import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord, make_cell, gridy2gridx_erp2pers, gridy2gridx_erp2fish, gridy2gridx_pers2erp, celly2cellx_erp2pers, celly2cellx_erp2fish, celly2cellx_pers2erp
from test_ltew import batched_predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--mode', default='erp2pers')
    parser.add_argument('--FOV')
    parser.add_argument('--THETA')
    parser.add_argument('--PHI')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    
    h, w = img.shape[-2:]
    H, W = list(map(int, args.resolution.split(',')))
    
    gridy = make_coord((H, W))
    if args.mode == 'erp2pers':
        gridx, mask = gridy2gridx_erp2pers(gridy, H, W, h, w, int(args.FOV), int(args.THETA), int(args.PHI))
    elif args.mode == 'erp2fish':
        gridx, mask = gridy2gridx_erp2fish(gridy, H, W, h, w, int(args.FOV), int(args.THETA), int(args.PHI))
    elif args.mode == 'pers2erp':
        gridx, mask = gridy2gridx_pers2erp(gridy, H, W, h, w)
    else:
        pass
    mask = mask.view(H, W, 1).permute(2, 0, 1).cpu()
    
    celly = make_cell(make_coord((H, W)), torch.ones(H, W))
    if args.mode == 'erp2pers':
        cellx = celly2cellx_erp2pers(celly, H, W, h, w, int(args.FOV), int(args.THETA), int(args.PHI))
    elif args.mode == 'erp2fish':
        cellx = celly2cellx_erp2fish(celly, H, W, h, w, int(args.FOV), int(args.THETA), int(args.PHI))
    elif args.mode == 'pers2erp':
        cellx = celly2cellx_pers2erp(celly, H, W, h, w)
    else:
        pass
    
    pred = batched_predict(model, ((img - 0.5) / 0.5).unsqueeze(0).cuda(),
        gridx.unsqueeze(0).cuda(), cellx.unsqueeze(0).cuda(), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp_(0, 1).view(H, W, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred*mask + 1-mask).save(args.output)