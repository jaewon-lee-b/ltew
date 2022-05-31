import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord, make_cell, gridy2gridx_erp2pers, celly2cellx_erp2pers
from test_lte import batched_predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
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
    gridx, mask = gridy2gridx_erp2pers(gridy, H, W, h, w, int(args.FOV), int(args.THETA), int(args.PHI))
    mask = mask.view(H, W, 1).permute(2, 0, 1).cpu()

    cell = make_cell(gridy, torch.ones(H, W))
    cell = celly2cellx_erp2pers(cell, H, W, h, w, int(args.FOV), int(args.FOV), int(args.THETA))
    
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        gridx.cuda().unsqueeze(0), cell.cuda().unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(H, W, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred*mask + 1-mask).save(args.output)