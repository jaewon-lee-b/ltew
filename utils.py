import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter

import cv2
from srwarp import transform

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_cell(coord, img):
    coord_bot_left  = coord + torch.tensor([-1/img.shape[-2], -1/img.shape[-1]]).unsqueeze(0)
    coord_bot_right = coord + torch.tensor([-1/img.shape[-2],  1/img.shape[-1]]).unsqueeze(0)
    coord_top_left  = coord + torch.tensor([ 1/img.shape[-2], -1/img.shape[-1]]).unsqueeze(0)
    coord_top_right = coord + torch.tensor([ 1/img.shape[-2],  1/img.shape[-1]]).unsqueeze(0)
    coord_left  = coord + torch.tensor([-1/img.shape[-2], 0]).unsqueeze(0)
    coord_right = coord + torch.tensor([ 1/img.shape[-2], 0]).unsqueeze(0)
    coord_bot   = coord + torch.tensor([ 0, -1/img.shape[-1]]).unsqueeze(0)
    coord_top   = coord + torch.tensor([ 0,  1/img.shape[-1]]).unsqueeze(0)

    cell_side   = torch.cat((coord_left, coord_right, coord_bot, coord_top), dim=0)
    cell_corner = torch.cat((coord_bot_left, coord_bot_right, coord_top_left, coord_top_right), dim=0)
    cell = torch.cat((cell_corner, cell_side, coord), dim=0)
    return cell


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        if isinstance(scale, list):
            valid = diff[..., shave[0]:-shave[0], shave[1]:-shave[1]]
        else:
            valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def calc_mpsnr(sr, hr, mask, dataset=None, rgb_range=1):
    diff = mask * (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            pass
        else:
            raise NotImplementedError
        valid = diff
    else:
        valid = diff
    mask_factor = sr.shape[-2]*sr.shape[-1]/torch.sum(mask)
    mse = valid.pow(2).mean()*mask_factor
    return -10 * torch.log10(mse)


def gridy2gridx_homography(gridy, H, W, h, w, m, cpu=True):
    # scaling
    gridy += 1
    gridy[:, 0] *= H / 2
    gridy[:, 1] *= W / 2
    gridy -= 0.5
    gridy = gridy.flip(-1)
    
    # coord -> homogeneous coord
    if cpu:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1)), dim=-1).double()
    else:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1).cuda()), dim=-1).double()
    
    # transform
    if cpu:
        m = transform.inverse_3x3(m)
    else:
        m = transform.inverse_3x3(m).cuda()
    gridx = torch.mm(m, gridy.permute(1, 0)).permute(1, 0)

    # homogeneous coord -> coord
    gridx[:, 0] /= gridx[:, -1]
    gridx[:, 1] /= gridx[:, -1]
    gridx = gridx[:, 0:2]

    # rescaling
    gridx = gridx.flip(-1)
    gridx += 0.5
    gridx[:, 0] /= h / 2
    gridx[:, 1] /= w / 2
    gridx -= 1
    gridx = gridx.float()

    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()
    
    return gridx, mask


def gridy2gridx_erp2pers(gridy, H, W, h, w, FOV, THETA, PHI):    
    # scaling    
    wFOV = FOV
    hFOV = float(H) / W * wFOV
    h_len = h*np.tan(np.radians(hFOV / 2.0))
    w_len = w*np.tan(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
    gridy = gridy.flip(-1)
    gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
    # project warped planed onto sphere
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    gridy /= hr_norm
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))
    
    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2]) / np.pi * 2
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) / np.pi
        
    gridx = torch.stack((lat, lon), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()
    
    return gridx, mask


def gridy2gridx_erp2fish(gridy, H, W, h, w, FOV, THETA, PHI):    
    # scaling    
    wFOV = FOV
    hFOV = float(H) / W * wFOV
    h_len = h*np.sin(np.radians(hFOV / 2.0))
    w_len = w*np.sin(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
    gridy = gridy.flip(-1)
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    
    mask = torch.where(hr_norm > 1, 0.0, 1.0)
    hr_norm = torch.where(hr_norm > 1, 1.0, hr_norm)
    hr_xaxis = torch.sqrt(1 - hr_norm**2)
    gridy = torch.cat((hr_xaxis, gridy), dim=-1)
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))

    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2].clamp_(-1+1e-6, 1-1e-6)) / np.pi * 2 # clamping to prevent arcsin explosion
    lon = torch.atan2(gridy[:, 1], gridy[:, 0]) / np.pi
    
    gridx = torch.stack((lat, lon), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = mask.squeeze(-1).float()
    
    return gridx, mask


def celly2cellx_homography(celly, H, W, h, w, m, cpu=True):
    cellx, _ = gridy2gridx_homography(celly, H, W, h, w, m, cpu) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_erp2pers(celly, H, W, h, w, FOV, THETA, PHI):
    cellx, _ = gridy2gridx_erp2pers(celly, H, W, h, w, FOV, THETA, PHI) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_erp2fish(celly, H, W, h, w, FOV, THETA, PHI):
    cellx, _ = gridy2gridx_erp2fish(celly, H, W, h, w, FOV, THETA, PHI) # backward mapping
    return shape_estimation(cellx)


def shape_estimation(cell):
    cell_1 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :] \
                - cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_2 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :] \
                - cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :] # Jacobian
    cell_3 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :] \
              - 2*cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] \
                + cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_4 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :] \
              - 2*cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] \
                + cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :] # Second-order derivatives in Hessian
    cell_5 = cell[3*cell.shape[0]//9:4*cell.shape[0]//9, :] \
                - cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] \
                - cell[1*cell.shape[0]//9:2*cell.shape[0]//9, :] \
                + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] \
                - cell[2*cell.shape[0]//9:3*cell.shape[0]//9, :] \
                + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] # Cross-term in Hessian
    shape = torch.cat((cell_1, cell_2, 4*cell_3, 4*cell_4, cell_5), dim=-1)
    return shape


def first_x_translation(a, b, e, f, g, x, y):
    num = a*f*y + a - e*(b*y + g)
    den = torch.square(f*y + e*x + 1)
    return num/den


def first_y_translation(a, b, e, f, g, x, y):
    num = -f*(a*x + g) + b + e*b*x
    den = torch.square(f*y + e*x + 1)
    return num/den


def second_x_translation(a, b, e, f, g, x, y):
    num = -2*e*(a*f*y + a - e*(b*y + g))
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def second_y_translation(a, b, e, f, g, x, y):
    num = -2*f*(-f*(a*x + g) + e*b*x + b)
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def second_xy_translation(a, b, e, f, g, x, y):
    num = -f*(a*f*y - e*a*x + a - 2*e*g) - e*b*(-f*y + e*x + 1)
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def JacobTInv(transf, x, y, h, w):
    a = transf[0, 0]
    b = transf[0, 1]
    c = transf[1, 0]
    d = transf[1, 1]
    e = transf[2, 0]
    f = transf[2, 1]
    g = transf[0, 2]
    h = transf[1, 2]

    JacobT = torch.zeros(2,2)
    JacobT[0, 0] = first_x_translation(a, b, e, f, g, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[1, 0] = first_x_translation(c, d, e, f, h, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[0, 1] = first_y_translation(a, b, e, f, g, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[1, 1] = first_y_translation(c, d, e, f, h, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobTInv = torch.linalg.inv(JacobT)
    return JacobTInv


def quantize(x: torch.Tensor) -> torch.Tensor:
    x = 127.5 * (x + 1)
    x = x.clamp(min=0, max=255)
    x = x.round()
    x = x / 127.5 - 1
    return x