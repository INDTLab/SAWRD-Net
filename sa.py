import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import nn as enn
from e2cnn import gspaces

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

def conv3x3(inplanes, planes, stride=1, padding=1, dilation=1, bias=False, fixparams=False):
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

def conv1x1(inplanes, planes, stride=1, padding=0, dilation=1, bias=False, fixparams=False):
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

def convnxn(inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, to_trivial=False, isGroups=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    if to_trivial:
        out_type = FIELD_TYPE['trivial'](gspace, planes, fixparams=False)
    else:
        out_type = FIELD_TYPE['regular'](gspace, planes)
    if cutoff_dilation and dilation > 1 and kernel_size == 3:
        frequencies_cutoff = lambda r: min(3*r, 3)
    else:
        frequencies_cutoff = lambda r: 3 * r
    if isGroups:
        groups = len(in_type)
    #print(f"len(in_type):{len(in_type)}")
    return enn.R2Conv(in_type, out_type, kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias,
                      dilation=dilation,
                      sigma=None,
                      frequencies_cutoff=frequencies_cutoff)

def build_norm_layer(num_features, postfix=''):
    in_type = FIELD_TYPE['regular'](gspace, num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)

def ennInnerBatchNorm(inplanes):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.InnerBatchNorm(in_type)

def ennReLU(inplanes, inplace=True):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.ReLU(in_type, inplace=inplace)

def ennDropout(inplanes, p):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseDropout(in_type, p)

def ennInterpolate(inplanes, scale_factor, mode='nearest', align_corners=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.R2Upsampling(in_type, scale_factor, mode=mode, align_corners=align_corners)

def ennPointwiseAvgPool(inplanes, kernel_size, stride=1, padding=0):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseAvgPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)

def AvgPool(inplanes):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseAdaptiveAvgPool(in_type, (1,1))

def MaxPool(inplanes):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseAdaptiveMaxPool(in_type, (1,1))

def build_eqnorm_layer(channel):
    in_type = FIELD_TYPE['regular'](gspace, channel)
    return enn.InnerBatchNorm(in_type)

def ennMaxPool(inplanes, kernel_size, stride=1, padding=0):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseMaxPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)

def ennPointwiseNonLinearity(inplanes):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseNonLinearity(in_type)

class SA_ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SA_ChannelAttention, self).__init__()
        self.avg = AvgPool(input_channels)
        self.maxpool = MaxPool(input_channels)
        self.fc1 = convnxn(inplanes=input_channels, planes=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.relu = ennReLU(internal_neurons)
        self.fc2 = convnxn(inplanes=internal_neurons, planes=input_channels, kernel_size=1, stride=1, bias=True)
        self.sigmoid = ennPointwiseNonLinearity(input_channels)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = self.avg(inputs)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.sigmoid(x1)
        x2 = self.maxpool(inputs)
        x2 = self.fc1(x2)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)
        x2 = self.sigmoid(x2)
        x = x1 + x2
        x = x.tensor
        x = x.view(-1, self.input_channels, 1, 1)
        inputs = inputs.tensor
        inputs = inputs * x
        in_type = FIELD_TYPE['regular'](gspace, inputs.shape[1])
        result = enn.GeometricTensor(inputs, in_type)
        return result

class SA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()
        self.ca = SA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.conv0 = convnxn(inplanes=channels,planes=channels,kernel_size=1,isGroups=True)
        self.conv0_1 = convnxn(inplanes=channels,planes=channels,kernel_size=3,padding=1,dilation=1,isGroups=True)
        self.conv1_1 = convnxn(inplanes=channels,planes=channels,kernel_size=3,padding=2,dilation=2,isGroups=True)
        self.conv2_1 = convnxn(inplanes=channels,planes=channels,kernel_size=3,padding=4,dilation=4,isGroups=True)
        self.conv = convnxn(channels, channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.act = ennReLU(channels)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        identity = inputs 
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        inputs = self.ca(inputs)
        x_init = self.conv0(inputs)
        x_1 = self.conv0_1(x_init)
        x_2 = self.conv1_1(x_init)
        x_3 = self.conv2_1(x_init)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        spatial_att = spatial_att.tensor
        inputs = inputs.tensor
        out = spatial_att * inputs
        out_type = FIELD_TYPE['regular'](gspace, out.shape[1])
        out = enn.GeometricTensor(out, out_type)
        out = identity + out
        out = self.conv(out)
        
        return out