# coding=utf-8
import pdb
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from PIL import Image
import numpy as np
from datasets import VOC, Saliency, VOCSELF
from datasets import palette as palette_voc
from evaluate_seg import evaluate_iou
from evaluate_sal import fm_and_mae
import json
import os
from jls_deeplab import JLSDL
from logger import Logger

image_size = 256
batch_size = 4
train_iters = 200000
c_output = 21
_num_show = 4
experiment_name = "debug2"
learn_rate = 5e-6
start_iter = 86000

path_save_valid_voc = "output/validation/{}_voc".format(experiment_name)
if not os.path.exists(path_save_valid_voc): os.mkdir(path_save_valid_voc)

path_save_valid_sal = "output/validation/{}_sal".format(experiment_name)
if not os.path.exists(path_save_valid_sal): os.mkdir(path_save_valid_sal)

path_save_checkpoints = "output/checkpoints/{}".format(experiment_name)
if not os.path.exists(path_save_checkpoints): os.mkdir(path_save_checkpoints)

net = JLSDL(c_output).cuda()
writer = Logger("output/logs/{}".format(experiment_name), 
        clear=True, port=8000, palette=palette_voc)


mean = torch.Tensor([0.485, 0.456, 0.406])[None, ..., None, None].cuda()
std = torch.Tensor([0.229, 0.224, 0.225])[None, ..., None, None].cuda()

voc_train_img_dir = './voc_train_pred_debug_img'
voc_train_plbl_dir = './voc_train_pred_debug_softcrf'
voc_train_gt_dir = './voc_train_pred_debug_gt'

voc_val_img_dir = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/JPEGImages'
voc_val_gt_dir = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClass'

voc_val_split = '/home/zeng/data/datasets/segmentation/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
voc_train_split = './trainself.txt'

sal_train_img_dir = '/home/zeng/data/datasets/saliency/DUT-train/images'
sal_train_gt_dir = '/home/zeng/data/datasets/saliency/DUT-train/masks'

sal_val_img_dir = '/home/zeng/data/datasets/saliency/ECSSD/images'
sal_val_gt_dir = '/home/zeng/data/datasets/saliency/ECSSD/masks'

sal_train_loader = torch.utils.data.DataLoader(
    Saliency(sal_train_img_dir, sal_train_gt_dir,
           crop=0.9, flip=True, rotate=10, size=image_size, training=True),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

sal_val_loader = torch.utils.data.DataLoader(
    Saliency(sal_val_img_dir, sal_val_gt_dir,
           crop=None, flip=False, rotate=None, size=image_size, training=False), 
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

voc_train_loader = torch.utils.data.DataLoader(
    VOCSELF(voc_train_img_dir, voc_train_gt_dir, voc_train_plbl_dir, voc_train_split,
           crop=None, flip=False, rotate=None, size=image_size, training=True),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

voc_val_loader = torch.utils.data.DataLoader(
    VOC(voc_val_img_dir, voc_val_gt_dir, voc_val_split,
           crop=None, flip=False, rotate=None, size=image_size, training=False),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

def val_sal():
    net.eval()
    with torch.no_grad():
        for it, (img, gt, batch_name, WW, HH) in tqdm(enumerate(sal_val_loader), desc='train'):
            img = (img.cuda()-mean)/std
            pred_seg, v_sal,_ = net(img)
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
            batch_seg, _, cls_fc = net(img)
            cls_fc = torch.sigmoid(cls_fc)
            #batch_seg[:, 1:] *= cls_fc[:, 1:][..., None, None]
            _, batch_seg = batch_seg.detach().max(1)
            for n, name in enumerate(batch_name):
                msk = batch_seg[n]
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


def train():
    print("============================= TRAIN ============================")

    voc_train_iter = iter(voc_train_loader)
    voc_it = 0
    sal_train_iter = iter(sal_train_loader)
    sal_it = 0
    log = {'best_miou': 0, 'best_it_miou': 0, 
            'best_mae': 1000, 'best_it_mae':0, 'best_fm':0, 'best_it_fm':0}
    optimizer = torch.optim.Adam([{'params': net.parameters(), 
        'lr': learn_rate, 'betas':(0.95, 0.999)}])
    if start_iter > 0:
        net.load_state_dict(torch.load(os.path.join(
            path_save_checkpoints, "{}.pth".format(start_iter))))

    for i in range(start_iter, train_iters):
        if i % 3000 == 1:
            _lr = learn_rate / float(2**(i//3000))
            optimizer = torch.optim.Adam([{'params': net.parameters(), 
                'lr': _lr, 'betas':(0.95, 0.999)}])
        """loss 1 """
        if sal_it >= len(sal_train_loader):
            sal_train_iter = iter(sal_train_loader)
            sal_it = 0
        img_sal, gt_sal = sal_train_iter.next()
        sal_it += 1
        img_sal_raw = img_sal
        gt_sal = gt_sal[:, None, ...].cuda()
        gt_sal = gt_sal.squeeze(1).long()
        img_sal_raw = img_sal
        img_sal = (img_sal.cuda()-mean)/std

        pred_seg, v_sal, _ = net(img_sal)
        pred_seg = torch.softmax(pred_seg, 1)
        bg = pred_seg[:, :1]
        fg = (pred_seg[:, 1:]*v_sal[:, 1:]).sum(1, keepdim=True)
        pred_sal = torch.cat((bg, fg), 1)
        loss_sal = F.nll_loss(pred_sal, gt_sal)

        """loss 2 """
        if voc_it >= len(voc_train_loader):
            voc_train_iter = iter(voc_train_loader)
            voc_it = 0
        img_seg, gt_seg, plbl_seg = voc_train_iter.next()
        voc_it += 1
        gt_cls = gt_seg[:, None, ...] == torch.arange(c_output)[None, ..., None, None]
        gt_cls = (gt_cls.sum(3).sum(2)>0).float().cuda()
        img_seg_raw = img_seg
        img_seg = (img_seg.cuda()-mean)/std
        pred_seg, _, cls_fc = net(img_seg)
        loss_cls = F.cross_entropy(pred_seg,plbl_seg.cuda())+\
                F.binary_cross_entropy_with_logits(cls_fc[:, 1:], gt_cls[:, 1:])
        loss = loss_cls+loss_sal

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """output """
        if i % 50 == 0:
            writer.add_scalar("sal_loss", loss_sal.item(), i)
            writer.add_scalar("cls_loss", loss_cls.item(), i)
            num_show = _num_show if img_seg.size(0) > _num_show else img_seg.size(0)
            img = img_seg_raw[-num_show:]
            writer.add_image('image_seg', torchvision.utils.make_grid(img), i)

            pred = plbl_seg[-num_show:,None,...]
            pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
            pred = pred[0]
            writer.add_label('gt_seg', pred,i)
            _, pred_label = pred_seg.max(1)
            pred = pred_label[-num_show:,None,...]
            pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
            pred = pred[0]
            writer.add_label('pred_seg', pred,i)

            img = img_sal_raw[-num_show:]
            writer.add_image('image_sal', torchvision.utils.make_grid(img), i)

            pred = gt_sal[-num_show:,None,...]
            pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
            pred = pred[0]
            writer.add_label('gt_sal', pred,i)

            pred = fg[-num_show:,...]
            pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
            writer.add_image('pred_sal', pred,i)
            writer.write_html()
            print("iter %d loss_sal %.4f; loss_cls %.4f"%(i, loss_sal.item(), loss_cls.item()))
        """validation"""
        if i!=0 and i % 500 == 0:
            log[i] = {}
            save_dict = net.state_dict()
            torch.save(save_dict, "{}/{}.pth".format(path_save_checkpoints, i))
            miou = val_voc()
            writer.add_scalar("miou", miou, i)
            log[i]['miou'] = miou
            if miou > log['best_miou']:
                log['best_miou'] = miou
                log['best_it_miou'] = i
            print("validation: iter %d; miou %.4f; best %d:%.4f"%(i, miou, log['best_it_miou'], log['best_miou']))
            fm, mae = val_sal()
            writer.add_scalar("mae", mae, i)
            writer.add_scalar("fm", fm, i)
            log[i]['mae'] = mae
            log[i]['fm'] = fm
            if mae < log['best_mae']:
                log['best_mae'] = mae
                log['best_it_mae'] = i
            if fm > log['best_fm']:
                log['best_fm'] = fm
                log['best_it_fm'] = i
            print("mae %.4f; best %d:%.4f"%(mae, log['best_it_mae'], log['best_mae']))
            print("fm %.4f; best %d:%.4f"%(fm, log['best_it_fm'], log['best_fm']))
            with open("output/{}.json".format(experiment_name), "w") as f:
                json.dump(log, f)




if __name__ == "__main__":
    train()
    #net.load_state_dict(torch.load("output/checkpoints/debug/500.pth"))
    #miou = val()
    #print(miou)
