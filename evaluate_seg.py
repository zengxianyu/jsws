import numpy as np
from PIL import Image
import os
import pdb
from multiprocessing import Pool
from functools import partial
from datetime import datetime

# ignore 255


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def evaluate(pred_dir, gt_dir, num_class):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    names = os.listdir(pred_dir)
    for name in names:
        pred = Image.open('{}/{}'.format(pred_dir, name)).convert('P')
        gt = Image.open('{}/{}'.format(gt_dir, name)).convert('P')
        pred = pred.resize(gt.size)
        pred = np.array(pred, dtype=np.int64)
        gt = np.array(gt, dtype=np.int64)
        gt[gt==255] = -1
        acc, pix = accuracy(pred, gt)
        intersection, union = intersectionAndUnion(pred, gt, num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))

    # print('[Eval Summary]:')
    # print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
    #       .format(iou.mean(), acc_meter.average() * 100))
    return iou.mean(), acc_meter.average()


def evaluate_one(pp, num_class):
    pred = Image.open(pp[0]).convert('P')
    gt = Image.open(pp[1]).convert('P')
    pred = pred.resize(gt.size)
    pred = np.array(pred, dtype=np.int64)
    gt = np.array(gt, dtype=np.int64)
    gt[gt == 255] = -1
    intersection, union = intersectionAndUnion(pred, gt, num_class)
    return [intersection, union]


def evaluate_iou(pred_dir, gt_dir, num_class):
    names = os.listdir(pred_dir)
    paths = [('{}/{}'.format(pred_dir, name), '{}/{}'.format(gt_dir, name)) for name in names]
    pool = Pool(4)
    results = pool.map(partial(evaluate_one, num_class=num_class), paths)
    results = np.array(results)
    ins = results[:, 0, :]
    uns = results[:, 1, :]
    # ins = []
    # uns = []
    # for pp in paths:
    #     intersection, union = evaluate_one(pp, num_class)
    #     ins += [intersection]
    #     uns += [union]
    # ins = np.array(ins)
    # uns = np.array(uns)
    iou = ins.sum(0) / (uns.sum(0)+1e-10)
    miou = iou.mean()
    # for i, _iou in enumerate(iou):
    #     print('class [{}], IoU: {}'.format(i, _iou))
    # print('[Eval Summary]:')
    # print('Mean IoU: {:.4}'
    #       .format(miou))
    return miou

    # for name in names:
    #     pred = Image.open('{}/{}'.format(pred_dir, name))
    #     gt = Image.open('{}/{}'.format(gt_dir, name)).convert('P')
    #     pred = pred.resize(gt.size)
    #     pred = np.array(pred, dtype=np.int64)
    #     gt = np.array(gt, dtype=np.int64)
    #     gt[gt==255] = -1
    #     intersection, union = intersectionAndUnion(pred, gt, num_class)
    #     intersection_meter.update(intersection)
    #     union_meter.update(union)
    # iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    # for i, _iou in enumerate(iou):
    #     print('class [{}], IoU: {}'.format(i, _iou))
    #
    # print('[Eval Summary]:')
    # print('Mean IoU: {:.4}'
    #       .format(iou.mean()))
    # return iou.mean()


if __name__ == "__main__":
    output_dir = '../WSLfiles/WT_densenet169/results'
    gt_dir = '../data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClass'
    miou = evaluate_iou(output_dir, gt_dir, 21)
    print(miou)
