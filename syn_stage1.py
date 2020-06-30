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
from jls_fcn import JLSFCN
from logger import Logger
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels
from multiprocessing import Pool

image_size = 256
batch_size = 8
c_output = 21
path_save_checkpoints = "stage1.pth"
path_save_train_voc = "voc_train_pred_debug"
path_save_train_voc_prob = "voc_train_pred_debug_prob"
path_save_train_voc_crf = "voc_train_pred_debug_softcrf"
voc_save_gt_dir = "voc_train_pred_debug_gt"
voc_save_img_dir = "voc_train_pred_debug_img"
if not os.path.exists(path_save_train_voc): os.mkdir(path_save_train_voc)
if not os.path.exists(path_save_train_voc_prob): os.mkdir(path_save_train_voc_prob)
if not os.path.exists(path_save_train_voc_crf): os.mkdir(path_save_train_voc_crf)
if not os.path.exists(voc_save_gt_dir): os.mkdir(voc_save_gt_dir)
if not os.path.exists(voc_save_img_dir): os.mkdir(voc_save_img_dir)

net = JLSFCN(c_output).cuda()
net.load_state_dict(torch.load(path_save_checkpoints))

mean = torch.Tensor([0.485, 0.456, 0.406])[None, ..., None, None].cuda()
std = torch.Tensor([0.229, 0.224, 0.225])[None, ..., None, None].cuda()

voc_train_img_dir = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/JPEGImages'
voc_train_gt_dir = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAug'
voc_train_split = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt'

voc_loader = torch.utils.data.DataLoader(
    VOC(voc_train_img_dir, voc_train_gt_dir, voc_train_split,
           crop=0.9, flip=True, rotate=10, size=image_size, training=False, tproc=True),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def val_voc():
    net.eval()
    with torch.no_grad():
        for t in range(10):
            for it, (img, gt, batch_name, WW, HH) in tqdm(enumerate(voc_loader), desc='train'):
                gt_cls = gt[:, None, ...] == torch.arange(c_output)[None, ..., None, None]
                gt_cls = (gt_cls.sum(3, keepdim=True).sum(2, keepdim=True)>0).float().cuda()
                raw_img = img
                img = (img.cuda()-mean)/std
                batch_seg, _, seg32x = net(img)
                batch_seg[:, 1:] *= gt_cls[:, 1:]
                batch_prob = F.softmax(batch_seg, 1)
                _, batch_seg = batch_seg.detach().max(1)
                for n, name in enumerate(batch_name):
                    msk = batch_seg[n]
                    prob = batch_prob[n]
                    prob = prob.detach().cpu().numpy()
                    np.save('{}/{}_{}.npy'.format(path_save_train_voc_prob, name, t), prob)

                    _img = raw_img[n]*255
                    _img = _img.numpy().astype(np.uint8)
                    _img = _img.transpose((1,2,0))
                    _img = Image.fromarray(_img)
                    _img.save('{}/{}_{}.jpg'.format(voc_save_img_dir, name, t))

                    _gt = gt[n]
                    _gt = _gt.numpy()
                    _gt = _gt.astype(np.uint8)
                    _gt = Image.fromarray(_gt.astype(np.uint8))
                    _gt = _gt.convert('P')
                    _gt.putpalette(palette_voc)
                    _gt.save('{}/{}_{}.png'.format(voc_save_gt_dir, name, t), 'PNG')

                    msk = msk.detach().cpu().numpy()
                    msk = Image.fromarray(msk.astype(np.uint8))
                    msk = msk.convert('P')
                    msk.putpalette(palette_voc)
                    msk.save('{}/{}_{}.png'.format(path_save_train_voc, name, t), 'PNG')
        miou = evaluate_iou(path_save_train_voc, voc_save_gt_dir, c_output)
        net.train()
        return miou

def one_dcrf(name):
    _name = ".".join(name.split(".")[:-1])
    img = Image.open(os.path.join(voc_save_img_dir, _name+".jpg")).convert("RGB")
    w,h = img.size
    img = np.array(img)
    prob = np.load(os.path.join(path_save_train_voc_prob, name))
    U = unary_from_softmax(prob)
    #msk = Image.open(os.path.join(path_save_train_voc, _name+".png")).convert("P")
    #msk = np.array(msk)
    #msk = msk+1
    #U = unary_from_labels(msk, c_output, gt_prob=0.7)
    d = dcrf.DenseCRF2D(w, h, c_output)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    msk = np.argmax(Q, axis=0).reshape((h,w))
    msk = msk
    msk = Image.fromarray(msk.astype(np.uint8))
    msk = msk.convert('P')
    msk.putpalette(palette_voc)
    msk.save('{}/{}.png'.format(path_save_train_voc_crf, _name), 'PNG')
    print(name)

def proc_dcrf():
    names = os.listdir(path_save_train_voc_prob)
    pool = Pool(4)
    pool.map(one_dcrf, names)
    #for name in tqdm(names):
    #    one_dcrf(name)
    miou = evaluate_iou(path_save_train_voc_crf, voc_save_gt_dir, c_output)
    return miou

if __name__ == "__main__":
    miou = val_voc()
    miou = proc_dcrf()
    # print(miou)
