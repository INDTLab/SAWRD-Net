import e2cnn.nn as enn
import math
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from e2cnn import gspaces
from mmengine.model import (constant_init, kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm


from sa import *

class _ASPPModule(enn.EquivariantModule):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.in_type = FIELD_TYPE['regular'](gspace, inplanes)
        self.out_type = FIELD_TYPE['regular'](gspace, planes)
        self.atrous_conv = convnxn(inplanes, planes, kernel_size, padding=padding, dilation=dilation, is_backbone=False)
        self.bn = enn.InnerBatchNorm(self.out_type)
        self.relu = ennReLU(planes)

    def forward(self, x):

        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        # assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape

    def export(self):
        for name, module in self._modules.items():
            if isinstance(module, enn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)
        return self

class _GAPModule(enn.EquivariantModule):
    def __init__(self, inplanes, planes):
        super(_GAPModule, self).__init__()
        self.in_type = FIELD_TYPE['regular'](gspace, inplanes)
        self.out_type = FIELD_TYPE['regular'](gspace, planes)
        self.global_avg_pool = enn.PointwiseAdaptiveAvgPool(self.in_type, (1, 1))
        self.conv = convnxn(inplanes, planes, kernel_size=1, is_backbone=False)
        self.bn = enn.InnerBatchNorm(self.out_type)
        self.relu = ennReLU(planes)

    def forward(self, x):

        x = self.global_avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)

        return self.relu(x)

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape

    def export(self):
        for name, module in self._modules.items():
            if isinstance(module, enn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)
        return self

class ASPP(nn.Module):
    def __init__(self, output_stride, in_channels):
        super(ASPP, self).__init__()
        inplanes = in_channels
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = _GAPModule(inplanes, 256)

        out_type = FIELD_TYPE['regular'](gspace, 256)
        self.conv1 = convnxn(256 * 5, 256, kernel_size=1, is_backbone=False)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu = ennReLU(256)
        self.dropout = enn.PointwiseDropout(out_type, 0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        scale_factor = (x4.shape[-2] / x5.shape[-2], x4.shape[-1] / x5.shape[-1])
        if torch.is_tensor(x5):
            upsample =  torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else: 
            upsample = enn.R2Upsampling(x5.type, scale_factor, mode='bilinear', align_corners=True)
        x5 = upsample(x5)
        if torch.is_tensor(x5):
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        else:
            x = enn.tensor_directsum((x1, x2, x3, x4, x5))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x #self.dropout(x)

    def export(self):
        for name, module in self._modules.items():
            if isinstance(module, enn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)
        return self