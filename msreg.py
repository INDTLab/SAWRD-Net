import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import nn as enn
from e2cnn import gspaces

import numpy as np
import matplotlib.pyplot as plt
import cv2
from mmcv.cnn import constant_init, kaiming_init


def set_gspace(n_angle, flip=True, cutoff_dilation=False):
    # Set default Orientation=8, .i.e, the group C8
    # One can change it by passing the env Orientation=xx
    Orientation = n_angle
    # keep similar computation or similar params
    # One can change it by passing the env fixparams=True
    fixparams = False
    if 'Orientation' in os.environ:
        Orientation = int(os.environ['Orientation'])
    if 'fixparams' in os.environ:
        fixparams = True
    #print('ReResNet Orientation: {}\tFix Params: {}\tCutoff Dilations:{}'.format(Orientation, fixparams, cutoff_dilation))

    if flip: # dihedral group
        #print('D%d group' % Orientation)
        gspace = gspaces.FlipRot2dOnR2(N=Orientation)
    else: # cyclic group
        #print('C%d group' % Orientation)
        gspace = gspaces.Rot2dOnR2(N=Orientation)

    return gspace, fixparams, cutoff_dilation

gspace, fixparams, cutoff_dilation = set_gspace(8,1,0)

def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = False):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()

    if fixparams:
        planes *= math.sqrt(N)

    planes = planes / N
    planes = int(planes)

    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = False):
    """ build a trivial feature map with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())

    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


def conv3x3(gspace, inplanes, planes, stride=1, padding=1, dilation=1, bias=False, fixparams=False):
    """3x3 convolution with padding"""
    in_type = FIELD_TYPE['regular'](gspace, inplanes, fixparams=fixparams)
    out_type = FIELD_TYPE['regular'](gspace, planes, fixparams=fixparams)
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)


def conv1x1(gspace, inplanes, planes, stride=1, padding=0, dilation=1, bias=False, fixparams=False):
    """1x1 convolution"""
    in_type = FIELD_TYPE['regular'](gspace, inplanes, fixparams=fixparams)
    out_type = FIELD_TYPE['regular'](gspace, planes, fixparams=fixparams)
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)

def convnxn(gspace, inplanes, planes, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1,  is_backbone=True, to_trivial=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    if to_trivial:
        out_type = FIELD_TYPE['trivial'](gspace, planes, fixparams=False)
    else:
        out_type = FIELD_TYPE['regular'](gspace, planes)
    if cutoff_dilation and is_backbone and dilation > 1 and kernel_size == 3:
        frequencies_cutoff = lambda r: min(3*r, 3)
    else:
        frequencies_cutoff = lambda r: 3 * r
    
    return enn.R2Conv(in_type, out_type, kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias,
                      dilation=dilation,
                      sigma=None,
                      frequencies_cutoff=frequencies_cutoff, )


def build_norm_layer(cfg, gspace, num_features, postfix=''):
    in_type = FIELD_TYPE['regular'](gspace, num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)

def ennInnerBatchNorm(gspace, inplanes):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.InnerBatchNorm(in_type)

def ennReLU(gspace, inplanes, inplace=True):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.ReLU(in_type, inplace=inplace)

def ennInterpolate(gspace, inplanes, scale_factor, mode='nearest', align_corners=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.R2Upsampling(in_type, scale_factor, mode=mode, align_corners=align_corners)

def ennPointwiseAvgPool2D(gspace, inplanes, kernel_size, stride=1, padding=0):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseMaxPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)


def ennMaxPool(gspace, inplanes, kernel_size, stride=1, padding=0):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseMaxPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)

class MSRE(enn.EquivariantModule):
    expansion = 4

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 dilation=1,
                 downsample=None, 
                 baseWidth=26, 
                 scale=4, 
                 stype='normal', 
                 gspace=None, 
                 fixparams=False):
        
        super(MSRE, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = convnxn(gspace, inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = ennInnerBatchNorm(gspace, width*scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        if stype == 'stage':
            self.pool = ennPointwiseAvgPool2D(gspace, inplanes, kernel_size=3, stride = stride, padding=1)
        
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(convnxn(
                gspace,
                width, 
                width,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                padding=dilation,
                bias=False))
            bns.append(ennInnerBatchNorm(gspace, width))
        self.stride=stride
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = convnxn(gspace, width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = ennInnerBatchNorm(gspace, planes * self.expansion)

        self.relu = ennReLU(gspace, width*scale)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        self.relu = ennReLU(gspace, out.shape[1])
        out = self.relu(out)

        spx = list(torch.split(out.tensor, self.width, 1))
        for i in range(self.nums):
            spxi_type = FIELD_TYPE['regular'](gspace, spx[i].shape[1])
            spx[i] = enn.GeometricTensor(spx[i], spxi_type)
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            self.relu = ennReLU(gspace, sp.shape[1])
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = enn.tensor_directsum((out, sp))

        spxnum_type = FIELD_TYPE['regular'](gspace, spx[self.nums].shape[1])
        spx[self.nums] = enn.GeometricTensor(spx[self.nums], spxnum_type)
        if self.scale != 1 and self.stype == 'normal':
            out = enn.tensor_directsum((out, spx[self.nums]))
        elif self.scale != 1 and self.stype == 'stage':
            self.pool = ennPointwiseAvgPool2D(gspace, spx[self.nums].shape[1], kernel_size=3, stride = self.stride, padding=1)
            out = enn.tensor_directsum((out, self.pool(spx[self.nums])))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        self.relu = ennReLU(gspace, out.shape[1])
        out = self.relu(out)

        return out
    
    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape


class MSREG(nn.Module):
    
    arch_settings = {
        168: (MSRE, (3, 4, 6, 3)),
        166: (MSRE, (3, 4, 6, 3)),
        164: (MSRE, (3, 4, 6, 3)),
        328: (MSRE, (3, 4, 6, 3)),
        326: (MSRE, (3, 4, 6, 3)),
        324: (MSRE, (3, 4, 6, 3)),
        488: (MSRE, (3, 4, 6, 3)),
        486: (MSRE, (3, 4, 6, 3)),
        484: (MSRE, (3, 4, 6, 3)),
        482: (MSRE, (3, 4, 6, 3)),
        648: (MSRE, (3, 4, 6, 3)),
        646: (MSRE, (3, 4, 6, 3)),
        644: (MSRE, (3, 4, 6, 3)),
        642: (MSRE, (3, 4, 6, 3)),
        808: (MSRE, (3, 4, 6, 3)),
        806: (MSRE, (3, 4, 6, 3)),
        804: (MSRE, (3, 4, 6, 3))
    }

    def __init__(self, 
                 depth,
                 baseWidth=26, 
                 scale=4, 
                 num_classes=1000,
                 orientation=8,
                 fixparams=False):
        self.inplanes = 64
        super(MSREG, self).__init__()
        self.blocks = [1, 2, 4]
        self.baseWidth = baseWidth
        self.scale = scale
        self.block, self.layers = self.arch_settings[depth]
        self.deep_stem = False
        self.gspace = gspaces.FlipRot2dOnR2(orientation)
        self.in_type = enn.FieldType(gspace, 3 * [gspace.trivial_repr])
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]

        self._make_stem_layer(self.gspace, 3, 64)

        self.layer1 = self._make_layer(self.gspace, self.block, 64, self.layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(self.gspace, self.block, 128, self.layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(self.gspace, self.block, 256, self.layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(self.gspace, self.block, 512, blocks=self.blocks, stride=strides[3], dilation=dilations[3])


    def _make_stem_layer(self, gspace, inplanes, stem_channels):
        if not self.deep_stem:
            in_type = enn.FieldType(
                gspace, inplanes * [gspace.trivial_repr])
            out_type = FIELD_TYPE['regular'](gspace, stem_channels)
            self.conv1 = enn.R2Conv(in_type, out_type, 7,
                                    stride=2,
                                    padding=3,
                                    bias=False,
                                    sigma=None,
                                    frequencies_cutoff=lambda r: 3 * r)
            self.bn1 = ennInnerBatchNorm(gspace, stem_channels)
            self.relu = enn.ReLU(self.conv1.out_type, inplace=True)
        self.maxpool = enn.PointwiseMaxPool(self.conv1.out_type, kernel_size=3, stride=2, padding=1)
        

    def _make_layer(self, gspace, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = enn.SequentialModule(
                convnxn(gspace, self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                ennInnerBatchNorm(gspace, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation,downsample=downsample, 
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale, gspace=gspace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,baseWidth=self.baseWidth, scale=self.scale, gspace=gspace))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, gspace, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = enn.SequentialModule(
                convnxn(gspace, self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                ennInnerBatchNorm(gspace, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=self.blocks[0]*dilation,downsample=downsample, 
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale, gspace=gspace))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,dilation=self.blocks[i]*dilation,
                                baseWidth=self.baseWidth, scale=self.scale, gspace=gspace))

        return nn.Sequential(*layers)
        
    def init_weights(self, pretrained=None):
        super(MSREG, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
    
    def forward(self, x):
        if not self.deep_stem:
            x = enn.GeometricTensor(x, self.in_type)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)

        return x
