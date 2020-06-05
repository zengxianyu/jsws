import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
import pdb
import random


def rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


class _BaseData(data.Dataset):
    def __init__(self, crop=None, rotate=None, flip=False):
        super(_BaseData, self).__init__()
        self.flip = flip
        self.rotate = rotate
        self.crop = crop

    def random_crop(self, *images):
        images = list(images)
        sz = [img.size for img in images]
        sz = set(sz)
        assert(len(sz)==1)
        w, h = sz.pop()
        th, tw = int(self.crop*h), int(self.crop*w)
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        results = [img.crop((j, i, j + tw, i + th)) for img in images]
        return tuple(results)

    def random_flip(self, *images):
        if self.flip and random.randint(0, 1):
            images = list(images)
            results = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            return tuple(results)
        else:
            return images

    def random_rotate(self, *images):
        images = list(images)
        sz = [img.size for img in images]
        sz = set(sz)
        assert(len(sz)==1)
        w, h = sz.pop()
        degree = random.randint(-1*self.rotate, self.rotate)
        images_r = [img.rotate(degree, expand=1) for img in images]
        w_b, h_b = images_r[0].size
        w_r, h_r = rotated_rect_with_max_area(w, h, np.radians(degree))
        ws = (w_b - w_r) / 2
        ws = max(ws, 0)
        hs = (h_b - h_r) / 2
        hs = max(hs, 0)
        we = ws + w_r
        he = hs + h_r
        we = min(we, w_b)
        he = min(he, h_b)
        results = [img.crop((ws, hs, we, he)) for img in images_r]
        return tuple(results)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class Saliency(_BaseData):
    def __init__(self, img_dir, gt_dir, img_format='jpg', gt_format='png', size=256, training=True,
                 crop=None, rotate=None, flip=False):
        super(Saliency, self).__init__(crop=crop, rotate=rotate, flip=flip)
        names = ['.'.join(name.split('.')[:-1]) for name in os.listdir(gt_dir)]
        self.img_filenames = [os.path.join(img_dir, name+'.'+img_format) for name in names]
        self.gt_filenames = [os.path.join(gt_dir, name+'.'+gt_format) for name in names]
        self.names = names
        self.size = size
        self.training = training

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # load image
        name = self.names[index]
        img_file = self.img_filenames[index]
        img = Image.open(img_file).convert("RGB")
        gt_file = self.gt_filenames[index]
        gt = Image.open(gt_file).convert("L")
        WW, HH = gt.size
        img = img.resize((WW, HH))
        if self.rotate is not None:
            img, gt = self.random_rotate(img, gt)
        if self.crop is not None:
            img, gt = self.random_crop(img, gt)
        if self.flip:
            img, gt = self.random_flip(img, gt)
        if self.size is not None:
            img = img.resize((self.size, self.size))
            gt = gt.resize((self.size, self.size))
        else:
            if min(w,h)<256:
                ratio = 256.0/min(w,h)
                w = int(ratio*w)
                h = int(ratio*h)
            w = (w//16+1)*16
            h = (h//16+1)*16
            img = img.resize((w,h))
            gt = gt.resize((w,h))

        img = np.array(img, dtype=np.float64) / 255.0
        gt = np.array(gt, dtype=np.uint8)
        gt = (gt>0)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).float()
        if self.training:
            return img, gt
        else:
            return img, gt, name, WW, HH

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


index2color = labelcolormap(21)
index2color = [list(hh) for hh in index2color]
index2name = ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
              'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')

class VOCSELF(_BaseData):
    def __init__(self, img_dir, gt_dir, self_dir, split_file, img_format='jpg', gt_format='png', size=256, training=True, crop=None, rotate=None, flip=False):
        super(VOCSELF, self).__init__(crop=crop, rotate=rotate, flip=flip)
        self.training = training
        self.size = size
        with open(split_file, 'r') as f:
            names = f.read().split('\n')[:-1]
        img_filenames = ['{}/{}.{}'.format(img_dir, name, img_format) for name in names]
        self.img_filenames = img_filenames
        self.names = names
        gt_filenames = ['{}/{}.{}'.format(gt_dir, _name, gt_format) for _name in names]
        self.gt_filenames = gt_filenames
        self_filenames = ['{}/{}.{}'.format(self_dir, _name, gt_format) for _name in names]
        self.self_filenames = self_filenames

    def __len__(self):
        return len(self.names)

    def train_proc(self, img, gt, plbl):
        data = (img, gt, plbl)
        if self.rotate is not None:
            data = self.random_rotate(*data)
        if self.crop is not None:
            data = self.random_crop(*data)
        if self.flip:
            data = self.random_flip(*data)
        img = data[0]
        gt = data[1]
        plbl = data[2]
        return img, gt, plbl

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file).convert("RGB")
        gt_file = self.gt_filenames[index]
        plbl_file = self.self_filenames[index]
        name = self.names[index]
        w, h = img.size
        gt = Image.open(gt_file).convert("P")
        plbl = Image.open(plbl_file).convert("P")
        img = img.resize(gt.size)
        if self.training:
            img, gt, plbl = self.train_proc(img, gt, plbl)
        if self.size is not None:
            img = img.resize((self.size, self.size))
            gt = gt.resize((self.size, self.size))
            plbl = plbl.resize((self.size, self.size))
        else:
            if min(w,h)<256:
                ratio = 256.0/min(w,h)
                w = int(ratio*w)
                h = int(ratio*h)
            w = (w//16+1)*16
            h = (h//16+1)*16
            img = img.resize((w,h))
            gt = gt.resize((w,h))
            plbl = plbl.resize((w,h))
        img = np.array(img, dtype=np.float64) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        gt = np.array(gt, dtype=np.int64)
        gt = torch.from_numpy(gt).long()
        plbl = np.array(plbl, dtype=np.int64)
        plbl = torch.from_numpy(plbl).long()
        if self.training:
            return img, gt, plbl
        else:
            return img, gt, plbl, name, w, h


class VOC(_BaseData):
    def __init__(self, img_dir, gt_dir, split_file, img_format='jpg', gt_format='png', size=256, training=True,
            crop=None, rotate=None, flip=False, tproc=False):
        super(VOC, self).__init__(crop=crop, rotate=rotate, flip=flip)
        self.training = training
        self.tproc = tproc
        self.size = size
        with open(split_file, 'r') as f:
            names = f.read().split('\n')[:-1]
        img_filenames = ['{}/{}.{}'.format(img_dir, name, img_format) for name in names]
        self.img_filenames = img_filenames
        self.names = names
        gt_filenames = ['{}/{}.{}'.format(gt_dir, _name, gt_format) for _name in names]
        self.gt_filenames = gt_filenames

    def __len__(self):
        return len(self.names)

    def train_proc(self, img, gt):
        data = (img, gt)
        if self.rotate is not None:
            data = self.random_rotate(*data)
        if self.crop is not None:
            data = self.random_crop(*data)
        if self.flip:
            data = self.random_flip(*data)
        img = data[0]
        gt = data[1]
        return img, gt

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file).convert("RGB")
        gt_file = self.gt_filenames[index]
        name = self.names[index]
        w, h = img.size
        gt = Image.open(gt_file).convert("P")
        img = img.resize(gt.size)
        if self.training or self.tproc:
            img, gt = self.train_proc(img, gt)
        if self.size is not None:
            img = img.resize((self.size, self.size))
            gt = gt.resize((self.size, self.size))
        else:
            if min(w,h)<256:
                ratio = 256.0/min(w,h)
                w = int(ratio*w)
                h = int(ratio*h)
            w = (w//16+1)*16
            h = (h//16+1)*16
            img = img.resize((w,h))
            gt = gt.resize((w,h))
        img = np.array(img, dtype=np.float64) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        gt = np.array(gt, dtype=np.int64)
        gt = torch.from_numpy(gt).long()
        if self.training:
            return img, gt
        else:
            return img, gt, name, w, h

if __name__ =="__main__":
    import pdb
    import matplotlib.pyplot as plt
    """
    path_img = "../data/datasets/saliency/DUT-train/images"
    path_mask = "../data/datasets/saliency/DUT-train/masks"
    dset = Saliency(path_img, path_mask, crop=0.9, rotate=10, flip=True) 
    img, gt = dset.__getitem__(0)
    img = img.numpy()
    gt = gt.numpy()
    img = img.transpose((1, 2, 0))
    plt.imshow(gt)
    plt.show()
    pdb.set_trace()
    """
    path_file_list = "../data/datasets/segmentation/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt"
    path_img = "../data/datasets/segmentation/VOCdevkit/VOC2012/JPEGImages"
    path_gt = "../data/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAug"
    dset = VOC(path_img, path_gt, path_file_list, crop=0.9, rotate=10, flip=True)
    img, gt = dset.__getitem__(0)
    img = img.numpy()
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()
    pdb.set_trace()
