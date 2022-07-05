import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale_lst = eval_type.split('-')
        if len(scale_lst) == 2: # symmetric-scale SR
            scale = int(scale_lst[1])
        elif len(scale_lst) == 3: # asymmetric-scale SR
            scale1 = int(math.ceil(float(scale_lst[1])))
            scale2 = int(math.ceil(float(scale_lst[2])))
            scale = [scale1, scale2]
        else:
            raise NotImplementedError
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        
        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            if not isinstance(scale, list): # symmetric-scale SR
                pred = batched_predict(model, inp, batch['coord'], batch['cell']*max(scale/scale_max, 1), eval_bsize) # cell clip for extrapolation
            else: # asymmetric-scale SR
                pred = batched_predict(model, inp, batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            if not isinstance(scale, list): # symmetric-scale SR
                # gt reshape
                ih, iw = batch['inp'].shape[-2:]
                s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
                shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
                batch['gt'] = batch['gt'].view(*shape) \
                    .permute(0, 3, 1, 2).contiguous()

                # prediction reshape
                shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
                pred = pred.view(*shape) \
                    .permute(0, 3, 1, 2).contiguous()
                pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
            else: # asymmetric-scale SR
                shape = [batch['inp'].shape[0], batch['target_h'].item(), batch['target_w'].item(), 3]
                pred = pred.view(*shape) \
                    .permute(0, 3, 1, 2).contiguous()
                batch['gt'] = batch['gt'].view(*shape) \
                    .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
            
    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    if args.model == 'bicubic':
        model = args.model
    else:
        model_spec = torch.load(args.model)['model']
        model = models.make(model_spec, load_sd=True).cuda()
        
    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        scale_max = int(args.scale_max),
        verbose=True)
    print('result: {:.4f}'.format(res))
