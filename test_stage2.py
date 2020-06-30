# coding=utf-8
import pdb
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from PIL import Image
import numpy as np
from datasets import VOC, Saliency
from datasets import palette as palette_voc
from evaluate_seg import evaluate_iou
from evaluate_sal import fm_and_mae
import json
import os
from jls_deeplab import JLSDL
from logger import Logger

image_size = 256
batch_size = 8
c_output = 21
experiment_name = "debug8"
path_save_checkpoints = "./output/checkpoints/debug8/105500.pth"

path_save_valid_voc = "output/validation/{}_voc".format(experiment_name)
if not os.path.exists(path_save_valid_voc): os.mkdir(path_save_valid_voc)

path_save_valid_sal = "output/validation/{}_sal".format(experiment_name)
if not os.path.exists(path_save_valid_sal): os.mkdir(path_save_valid_sal)

net = JLSDL(c_output).cuda()
net.load_state_dict(torch.load(path_save_checkpoints))

mean = torch.Tensor([0.485, 0.456, 0.406])[None, ..., None, None].cuda()
std = torch.Tensor([0.229, 0.224, 0.225])[None, ..., None, None].cuda()

voc_val_img_dir = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/JPEGImages'
voc_val_gt_dir = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClass'
voc_val_split = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

sal_val_img_dir = '/home/zeng/data/datasets/saliency/ECSSD/images'
sal_val_gt_dir = '/home/zeng/data/datasets/saliency/ECSSD/masks'

sal_val_loader = torch.utils.data.DataLoader(
    Saliency(sal_val_img_dir, sal_val_gt_dir,
           crop=None, flip=False, rotate=None, size=image_size, training=False), 
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

voc_val_loader = torch.utils.data.DataLoader(
    VOC(voc_val_img_dir, voc_val_gt_dir, voc_val_split,
           crop=None, flip=False, rotate=None, size=image_size, training=False),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def val_sal():
    net.eval()
    with torch.no_grad():
        for it, (img, gt, batch_name, WW, HH) in tqdm(enumerate(sal_val_loader), desc='train'):
            img = (img.cuda()-mean)/std
            pred_seg, v_sal, _ = net(img)
            pred_seg = torch.softmax(pred_seg, 1)
            bg = pred_seg[:, :1]
            fg = (pred_seg[:, 1:]*v_sal[:, 1:]).sum(1, keepdim=True)
            fg = fg.squeeze(1)
            fg = fg*255
            for n, name in enumerate(batch_name):
                msk =fg[n]
                msk = msk.detach().cpu().numpy()
                w = WW[n]
                h = HH[n]
                msk = Image.fromarray(msk.astype(np.uint8))
                msk = msk.resize((w, h))
                msk.save('{}/{}.png'.format(path_save_valid_sal, name), 'PNG')
        fm, mae, _, _ = fm_and_mae(path_save_valid_sal, sal_val_gt_dir)
        net.train()
        return fm, mae


def val_voc():
    net.eval()
    with torch.no_grad():
        for it, (img, gt, batch_name, WW, HH) in tqdm(enumerate(voc_val_loader), desc='train'):
            img = (img.cuda()-mean)/std
            outputs = net(img)
            batch_seg = outputs[0]
            _, batch_seg = batch_seg.detach().max(1)
            for n, name in enumerate(batch_name):
                msk =batch_seg[n]
                msk = msk.detach().cpu().numpy()
                w = WW[n]
                h = HH[n]
                msk = Image.fromarray(msk.astype(np.uint8))
                msk = msk.convert('P')
                msk.putpalette(palette_voc)
                msk = msk.resize((w, h))
                msk.save('{}/{}.png'.format(path_save_valid_voc, name), 'PNG')
        miou = evaluate_iou(path_save_valid_voc, voc_val_gt_dir, c_output)
        net.train()
        return miou

if __name__ == "__main__":
    fm, mae = val_sal()
    print(fm, mae)
    #net.load_state_dict(torch.load("output/checkpoints/debug/500.pth"))
    #miou = val_voc()
    #print(miou)
