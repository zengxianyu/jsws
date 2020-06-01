import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

#from densenet import *
from torchvision.models import densenet169


import numpy as np
import sys
thismodule = sys.modules[__name__]
import pdb

class Pass(nn.Module):
    def forward(self, x):
        return x

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.ConvTranspose2d) and m.in_channels == m.out_channels:
        initial_weight = get_upsampling_weight(
            m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(initial_weight)


def fraze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.requires_grad=False


def proc_densenet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features.transition3[-2].register_forward_hook(hook)
    model.features.transition2[-2].register_forward_hook(hook)

    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)
    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock4)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size==(3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)
    model.classifier = None
    model.forward = model.features.forward
    return model


def proc_vgg(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features[3][-2].register_forward_hook(hook)
    model.features[2][-2].register_forward_hook(hook)
    model.classifier = None
    return model

dim_dict = {
    'densenet169': [64, 128, 256, 640, 1664],
    'vgg16': [64, 128, 256, 512, 512],
    'mobilenet2': [32, 24, 32, 64, 1280],
    'resnet101': [64, 256, 512, 1024, 2048]
}


procs = {'densenet169': proc_densenet}



class JLSFCN(nn.Module):
    def __init__(self, c_output=21, base='densenet169'):
        super(JLSFCN, self).__init__()
        dims = dim_dict[base][::-1]
        self.pred_seg = nn.ModuleList([
            nn.Conv2d(dims[0], c_output, 1),
            nn.Conv2d(dims[1], c_output, 1),
            nn.Conv2d(dims[2], c_output, 1)])
        self.pred_sal = nn.Conv2d(dims[0], c_output, kernel_size=16)
        self.upsample = nn.ModuleList([
            #nn.ConvTranspose2d(c_output, c_output, kernel_size=4, stride=2, padding=1),
            Pass(),
            nn.ConvTranspose2d(c_output, c_output, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(c_output, c_output, kernel_size=16, stride=8, padding=4)])
        self.apply(weight_init)
        self.feature = getattr(thismodule, base)(pretrained=True)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)
        #self.apply(fraze_bn)

    def forward(self, x, boxes=None, ids=None):
        self.feature.feats[x.device.index] = []
        feat32 = self.feature(x)
        feat8, feat16 = self.feature.feats[x.device.index]
        seg32 = self.pred_seg[0](feat32)
        seg32x = seg32
        seg32 = self.upsample[0](seg32)
        seg16= self.pred_seg[1](feat16)
        seg16 = seg16+seg32
        seg16 = self.upsample[1](seg16)
        seg8 = self.pred_seg[2](feat8)
        seg =seg16+seg8

        seg = self.upsample[2](seg)
        sal = self.pred_sal(feat32)
        sal = sal.mean(3,keepdim=True).mean(2, keepdim=True)
        sal = torch.sigmoid(sal)
        return seg, sal, seg32x


if __name__ == "__main__":
    fcn = JLSFCN(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
    pdb.set_trace()
