import functools
import random
import math
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples, make_coord, make_cell, gridy2gridx_homography, celly2cellx_homography

from srwarp import transform

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
    
    
@register('sr-asp-implicit-paired')
class SRAspImplicitPaired(Dataset):

    def __init__(self, dataset, scale=1, scale2=2):
        self.dataset = dataset
        self.scale = scale
        self.scale2 = scale2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        scale = self.scale
        scale2 = self.scale2
        
        img_lr, img_hr = self.dataset[idx]               

        C, H_lr, W_lr = img_lr.size()
        C, H_hr, W_hr = img_hr.size()
        H = H_lr if round(H_lr * scale) <= H_hr else math.floor(H_hr / scale)
        W = W_lr if round(W_lr * scale2) <= W_hr else math.floor(W_hr / scale2)

        step = []
        for s in [scale, scale2]:
            if s == int(s):
                step.append(1)
            elif s * 2 == int(s * 2):
                step.append(2)
            elif s * 5 == int(s * 5):
                step.append(5)
            elif s * 10 == int(s * 10):
                step.append(10)
            elif s * 20 == int(s * 20):
                step.append(20)
            elif s * 50 == int(s * 50):
                step.append(50)

        H_new = H // step[0] * step[0]
        if H_new % 2 == 1:
            H_new = H // (step[0] * 2) * step[0] * 2

        W_new = W // step[1] * step[1]
        if W_new % 2 == 1:
            W_new = W // (step[1] * 2) * step[1] * 2

        img_lr = img_lr[:, :H_new, :W_new]
        img_hr = img_hr[:, :round(scale * H_new), :round(scale2 * W_new)]
        
        crop_hr, crop_lr = img_hr, img_lr
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
    
        target_h = crop_hr.shape[-2]
        target_w = crop_hr.shape[-1]
    
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'target_h': target_h,
            'target_w': target_w,
        }
    
    
@register('sr-implicit-homography-paired')
class SRImplicitHomographyPaired(Dataset):

    def __init__(self, dataset, scaling=False, max_scale=1/0.35, scale=None):
        self.dataset = dataset
        self.scaling = scaling
        self.max_scale = max_scale
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr, m = self.dataset[idx]
        
        h, w = img_lr.shape[-2:]
        H, W = img_hr.shape[-2:]
        
        # grid_x & mask
        coord_y, _ = to_pixel_samples(img_hr.contiguous())
        coord_x, mask = gridy2gridx_homography(coord_y.clone(), H, W, h, w, m) # backward mapping
        mask = mask.view(H, W, 1).permute(2, 0, 1)

        # cell
        cell = make_cell(coord_y, img_hr)
        
        if self.scaling:
            cell = celly2cellx_homography(cell, H, W, h, w, transform.scaling(min(self.scale, self.max_scale)).squeeze(0)) # clipping for extrapolation as in LTE
        else:
            cell = celly2cellx_homography(cell, H, W, h, w, m) # backward mapping
        
        return {
            'inp'  : img_lr,
            'coord': coord_x,
            'cell' : cell,
            'gt'   : img_hr,
            'mask' : mask
        }
    
    
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

    
@register('sr-asp-implicit-downsampled')
class SRAspImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s1 = random.uniform(self.scale_min, self.scale_max)
        s2 = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s1 + 1e-9)
            w_lr = math.floor(img.shape[-1] / s2 + 1e-9)
            img = img[:, :round(h_lr * s1), :round(w_lr * s2)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            h_hr = round(w_lr * s1)
            w_hr = round(w_lr * s2)
            x0 = random.randint(0, img.shape[-2] - h_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + h_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, (w_lr, w_lr))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }    

    
@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

    
@register('warp-yspace-coord-cell')
class WarpYspaceCoordCell(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # crop image for batching
        img = self.dataset[idx]
        x0 = random.randint(0, img.shape[-2] - self.inp_size)
        y0 = random.randint(0, img.shape[-1] - self.inp_size)
        img = img[:, x0: x0 + self.inp_size, y0: y0 + self.inp_size]        
        
        # augmentation
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img = augment(img)
        
        # prepare coordinate in Y space
        gridy, hr_rgb = to_pixel_samples(img.contiguous())
        
        # prepare cell in Y space
        cell = make_cell(gridy, img)
        
        return {
            'inp': img,
            'cell': cell,
            'coord': gridy,
            'gt': hr_rgb
        } 