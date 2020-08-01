import torch
import torch.nn as nn
import math
from .CommNet import *


class MoSliceNet(nn.Module):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__()
        self.in_chs = 32
        self.num_feature = 1280
        self.reorg = Reorg(slice)
        self.slice = slice
        self.conv_stem = conv3x3(3, self.in_chs, 2, slice=self.slice)
        self.bn1 = norm_layer(self.in_chs, slice=self.slice)
        self.act1 = act_layer()

        
        self.blocks = self.make_blocks(num_blocks, out_channels, strides, ch_multi, depth_multi)
        
        self.conv_head = conv1x1(self.in_chs, self.num_feature, slice=self.slice)
        self.bn2 = norm_layer(self.num_feature, self.slice)
        self.act2 = act_layer()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_feature, num_class)

        self.weight_init()

    def forward(self, x):
        x = self.reorg(x)
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        return nn.Identity()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)


class MoSliceNetV1(MoSliceNet):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)
        self.in_chs = 32

        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi))
        self.blocks = nn.Sequential(*blocks)

        self.weight_init()


    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if slice < 0:
            slice = abs(slice)
            self.in_chs = self.in_chs * slice *slice
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0]/slice/slice))
            slice = 1
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)

def moslicenet():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MoSliceNetV1(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)

class MoSliceNetV2(MoSliceNet):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice=1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   
        self.in_chs = 32
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi))
        self.blocks = nn.Sequential(*blocks)
        
        self.weight_init()

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if slice < 0:
            slice = abs(slice)
            self.in_chs = self.in_chs * slice *slice
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0]/slice/slice))
            slice = 1
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv2():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MoSliceNetV2(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)



class MoSliceNetV3(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   
        self.in_chs = 32

        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, slice=[slice//2, slice//2]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi, slice=[1, 1]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[8], out_channels[8], strides[8], ch_multi))
        self.blocks = nn.Sequential(*blocks)
        self.conv_head = conv1x1(self.in_chs, self.num_feature)

        self.weight_init()

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if isinstance(slice, list):
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            self.in_chs = out_chs
            if slice[0]==1:
                layers.append(AddSlice())
                slice = 1
            else:
                layers.append(FuseMulti(slice=slice))
                slice = [1, 1]  
        elif slice < 0:
            slice = abs(slice)
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            layers.append(Fuse(slice=slice))
            slice = [slice//2, slice//2]
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv3():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV3(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)


class MoSliceNetV3_2(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        slice = self.slice
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, slice=[slice//2, slice//2]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi, slice=[1, 1]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[8], out_channels[8], strides[8], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if isinstance(slice, list):
            if slice[0]==1:
                layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=1, real_in_chs=self.in_chs * 2))
                # layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
                # layers.append(AddSlice())
                slice = 1
                self.slice = slice
            else:
                # layers.append(FuseDiffV3(self.in_chs, out_chs, [stride, 1], exp_ratio=exp_ratios[0], slice=slice))
                layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
                layers.append(FuseMulti(slice=slice))
                slice = [1, 1]  
                self.slice = slice
        elif slice < 0:
            slice = abs(slice)
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            layers.append(Fuse(slice=slice))
            slice = [slice//2, slice//2]
            self.slice = slice
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv3_2():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV3_2(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)


class MoSliceNetV3_3(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        slice = self.slice
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, slice=-slice//2))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi, slice=1))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[8], out_channels[8], strides[8], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if slice < 0:
            slice = abs(slice)
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            layers.append(FuseSingle(slice=slice))
            slice = slice // 2
            self.slice = slice
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv3_3():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV3_3(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)


class MoSliceNetV3_6(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        slice = self.slice
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, slice=1))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if slice < 0:
            slice = abs(slice)
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            layers.append(FuseSingle(slice=slice))
            slice = slice // 2
            self.slice = slice
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv3_6():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 3, 1]
    return MoSliceNetV3_6(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=2)

class MoSliceNetV3_4(MoSliceNetV3_3):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice) 
        self.reorg = ReorgOverlap2(patch=64, overlap=12)

def moslicenetv3_4():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV3_4(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)


class MoSliceNetV4(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   
        self.in_chs = 32

        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, slice=[slice//2, slice//2]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi, slice=[slice//2, 1, 1])) #slice=[slice//2, 1]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[8], out_channels[8], strides[8], ch_multi, slice=slice//2))
        self.blocks = nn.Sequential(*blocks)
        self.conv_head = conv1x1(self.in_chs, self.num_feature, slice=slice//2)
        self.bn2 = norm_layer(self.num_feature, slice=slice//2)
        self.global_pool =  nn.Sequential(InvertedReorg(slice=slice//2), self.global_pool)
        
        self.weight_init()

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if isinstance(slice, list):
            if slice[1]==1:
                layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
                self.in_chs = out_chs
                layers.append(AddSliceDiffV2(slice=slice))
                slice = 2
            else:
                layers.append(FuseDiffV2(self.in_chs, out_chs, stride, exp_ratios=exp_ratios, slice=slice))
                slice = [2, 1, 1] #slice = [2, 1]  
        elif slice < 0:
            slice = abs(slice)
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            layers.append(Fuse(slice=slice))
            slice = [slice//2, slice//2]
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv4():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV4(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)


class MoSliceNetV5(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   
        self.in_chs = 32

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        slice = self.slice
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, slice=-(slice-1)))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi, slice=-(slice-2)))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[8], out_channels[8], strides[8], ch_multi))
        return nn.Sequential(*blocks)


    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if slice < 0:
            slice = abs(slice)
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            layers.append(FuseOverlap(slice=slice, kernel=2))
            slice = slice - 1
            self.slice = slice
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)

def moslicenetv5():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 2, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV5(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)


class MoSliceNetV6(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = [4, 4]):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   
        self.global_pool =  nn.Sequential(InvertedReorg(slice=self.slice), self.global_pool)

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, ops="normal")]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[2.0, 4.0], ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, exp_ratios=[4.0, 6.0], ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, ops="fuse"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi, ops="add"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[8], out_channels[8], strides[8], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], ops="normal"):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if ops == "normal":
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
        elif ops == "add":
                layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
                layers.append(AddSlice())
                self.slice = self.slice[0]
        elif ops == "fuse":
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
            layers.append(FuseMultiReorg(slice=self.slice, reorg = [self.slice[0]//2, self.slice[1]//2]))
            self.slice = [self.slice[0]//2, self.slice[1]//2]

        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=self.slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv6():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV6(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=[4, 4])


class MoSliceNetV6_2(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = [4, 4]):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   
        self.global_pool =  nn.Sequential(InvertedReorg(slice=self.slice), self.global_pool)

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, ops="normal")]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[2.0, 4.0], ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, exp_ratios=[4.0, 6.0], ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, ops="fuse"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi, ops="add"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[8], out_channels[8], strides[8], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], ops="normal"):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if ops == "normal":
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
        elif ops == "add":
                layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
                layers.append(AddSlice())
                self.slice = self.slice[0]
        elif ops == "fuse":
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
            layers.append(FuseMultiReorg(slice=self.slice, reorg = [self.slice[0]//2, self.slice[1]//2]))
            self.slice = [self.slice[0]//2, self.slice[1]//2]

        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=self.slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv6_2():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV6_2(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=[4, 4])


class MoSliceNetV7(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)
        # self.reorg = ReorgOverlap(slice=slice)   
        self.global_pool =  nn.Sequential(InvertedReorg(self.slice), self.global_pool)

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, ops="fuse"))
        # blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, ops="fuse"))
        # blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[-1], out_channels[-1], strides[-1], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], ops="normal"):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if ops == "fuse":
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
            layers.append(FuseSingle(slice=self.slice))
            self.slice = self.slice // 2
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=self.slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv7():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MoSliceNetV7(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)


class MoSliceNetV8(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   
        self.global_pool = PoolMulti(self.slice[0])
        self.classifier = nn.Linear(self.num_feature * 2, num_class)


    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, ops="fuse"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, ops="split"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[-1], out_channels[-1], strides[-1], ch_multi))
        return nn.Sequential(*blocks)
        
    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], ops="normal"):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if ops == "fuse":
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
            layers.append(FuseSingle(slice=self.slice))
            self.slice = self.slice // 2
        elif ops == "split":
            layers.append(FuseDiffV4(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
            self.slice = [2, 1]
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=self.slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv8():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV8(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)




class MoSliceNetV9(MoSliceNet):
    '''
    16*(64-32-26-8-4-2)/8*(4)/2*(4-1)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = [4, 4]):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)  
        self.reorg = ReorgPyramid(slice=slice) 

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, ops="normal")]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[2.0, 4.0], ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, exp_ratios=[4.0, 6.0], ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, ops="fuse"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi, ops="reduce"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi, depth_multi, ops="cat"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[8], out_channels[8], strides[8], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], ops="normal"):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if ops == "fuse":
            layers.append(FusePyramid(self.in_chs, out_chs, stride=[stride, stride, 1], exp_ratio=exp_ratios[0], slice=self.slice))
            self.slice = [self.slice[0]//2, self.slice[1]//2, self.slice[2]]
        elif ops == "cat":
            real_in_chs = self.in_chs // 16 + self.in_chs // 4 + self.in_chs
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=1, real_in_chs=real_in_chs ))
        elif ops == "reduce":
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[0], slice=self.slice))
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
        
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=self.slice))
            self.in_chs = out_chs
        if ops == "reduce":
            layers.pop()
            layers.append(ReducePyramid(self.in_chs, out_chs, stride=[stride, 1, 1], exp_ratio=exp_ratios[0], slice=self.slice))
            self.slice = 1
        return nn.Sequential(*layers)


def moslicenetv9():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV9(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=[4, 2, 1])


class MoSliceNetV9_3(MoSliceNetV9):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = [4, 4]):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)  
        
    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], ops="normal"):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if ops == "fuse":
            layers.append(FusePyramid(self.in_chs, out_chs, stride=[stride, stride, 1], exp_ratio=exp_ratios[0], slice=self.slice))
            self.slice = [self.slice[0]//2, self.slice[1]//2, self.slice[2]]
        elif ops == "cat":
            # real_in_chs = self.in_chs // 16 + self.in_chs // 4 + self.in_chs
            layers.append(AddSlice(patches=3))
            self.slice = 1
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
        elif ops == "reduce":
            layers.append(ReducePyramid3(self.in_chs, out_chs, stride=[stride, 1, 1], exp_ratio=exp_ratios[0], slice=self.slice))
            self.slice = [1, 1, 1]
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
        
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=self.slice))
            self.in_chs = out_chs
            
        return nn.Sequential(*layers)

def moslicenetv9_3():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 2, 1]
    return MoSliceNetV9_3(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=[4, 2, 1])



class MoSliceNetV10(MoSliceNet):
    '''
    4*(64-32-26-8-4-2)/1*(4)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice) 

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        slice = self.slice
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if slice < 0:
            slice = abs(slice)
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            layers.append(FuseSingle(slice=slice))
            slice = slice // 2
            self.slice = slice
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
        self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv10():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MoSliceNetV10(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=2)

class MoSliceNetV10_2(MoSliceNetV10):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)   

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        slice = self.slice
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi))
        return nn.Sequential(*blocks)
    

def moslicenetv10_2():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 1]
    return MoSliceNetV10_2(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=2)


class MoSliceNetV10_3(MoSliceNet):
    '''

    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = [4, 4]):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice)  
        self.reorg = ReorgPyramid2(slice=slice) 
        self.global_pool = PoolMulti2(self.slice[0])
        self.classifier = nn.Linear(self.num_feature * 2, num_class)  

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, ops="normal")]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[2.0, 4.0], ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, exp_ratios=[4.0, 6.0], ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, ops="normal"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, ops="fuse"))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[7], out_channels[7], strides[7], ch_multi))
        return nn.Sequential(*blocks)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], ops="normal"):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if ops == "fuse":
            layers.append(FusePyramid2(self.in_chs, out_chs, stride=[stride, 1], exp_ratio=exp_ratios[0], slice=self.slice))
            self.slice = [1, 1]
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=self.slice))
        
        self.in_chs = out_chs
        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=self.slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)


def moslicenetv10_3():
    output_channels = [16, 24, 32, 64, 96, 160, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 2, 1]
    return MoSliceNetV10_3(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=[2, 1])

class MoSliceNetV10_4(MoSliceNetV10):
    '''
    4*(64-32-26-8-4-2)/1*(4)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice) 

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        slice = self.slice
        blocks = [self.make_layers(DepthwiseConvRelu, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidualRelu, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidualRelu, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidualRelu, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidualRelu, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidualRelu, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidualRelu, num_blocks[6], out_channels[6], strides[6], ch_multi))
        return nn.Sequential(*blocks)

def moslicenetv10_4():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MoSliceNetV10_4(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=2)

class MoSliceNetV10_5(MoSliceNetV10):
    '''
    4*(64-32-26-8-4-2)/1*(4)/1*1
    '''
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi, slice=slice) 

    def make_blocks(self, num_blocks, out_channels, strides, ch_multi, depth_multi):
        slice = self.slice
        blocks = [self.make_layers(DepthwiseConvNoRes, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidualNoRes, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidualNoRes, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidualNoRes, num_blocks[3], out_channels[3], strides[3], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidualNoRes, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, slice=slice))
        blocks.append(self.make_layers(InvertedResidualNoRes, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidualNoRes, num_blocks[6], out_channels[6], strides[6], ch_multi))
        return nn.Sequential(*blocks)
        
def moslicenetv10_5():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MoSliceNetV10_5(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=2)
    
if __name__ == "__main__":
    # net = Reorg(slice=4)
    # x = torch.range(0, 191).reshape(1, 3, 8, 8)
    # print(x[0, 1, :, :])
    # x = net(x)
    # print(x.shape)
    # # print(x[0, ::3, :, :])

    # net = FuseOverlap(slice=4, kernel=2)
    # x = net(x)
    # print(x.shape)
    # print(x[0, 1::3, :, :])
    
    # net2 = InvertedReorg(slice=4)
    # x = net2(x)
    # print(x[0, 0, :, :])
    # x = Reshuffle()(x)
    # x = Reorg(slice=2)(x)
    # print(x[0, ::3, :, :])

    # fuse = Fuse(slice=4)
    # x = fuse(x)
    # print(x[0, 0, :, :])
    # print(x[0, ::3, :, :])
    # fuse2 = FuseMulti(slice=2)
    # x = fuse2(x)
    # print(x[0, ::3, :, :])
    # x = AddSlice()(x)
    # print(x[0, 0, :, :])

    # model = moslicenetv3()
    # print(model)

    from torchsummary import summary
    import torch.backends.cudnn as cudnn
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    model =  moslicenetv10_3().cuda()
    print(model)
    cudnn.benchmark = True
    summary(model, (3, 128, 128))


    # import torch
    # inputs = torch.arange(0, 12).reshape((1, 12, 1, 1)).float()
    # conv1 = conv1x1(3, 4, slice=2)
    # conv1.weight = torch.nn.Parameter(torch.arange(0, 16).reshape(16, 1, 1, 1).expand(16, 3, 1, 1).float())
    # print(conv1.weight)
    # print(conv1(inputs))

    # x = torch.range(0, 256*48-1).reshape(1, 1, 480, 256).repeat(1, 3, 4, 1)
    # net = ReorgOverlap(slice=4, overlap=1)
    # x = net(x)
    # print(x.shape)
    # print(x[0, :18:3, :, :])
     

    # print(generate_grid(256, 4, 1))
    
    # x = torch.range(1, 20).reshape(1, 5, 2, 2)
    # net = AddSliceDiff(slice=[2, 1])
    # x = net(x)
    # print(x)


    # x = torch.range(0, 191).reshape(1, 3, 8, 8)
    # print(x[0, 0, :, :])
    # net = ReorgPw(stride=4)
    # x = net(x)
    # print(x.shape)
    # print(x[0, ::3, :, :])

    # import torch
    # from torch.utils.tensorboard import SummaryWriter
    # from torch.autograd import Variable
    # dummy_input = Variable(torch.rand(1, 3, 256, 256)) #假设输入1张1*28*28的图片
    # model = moslicenetv6()
    # with SummaryWriter(comment='mobile_darwinnet') as w:
    #     w.add_graph(model, (dummy_input, ))


''' moslicenet_apple
class MoSliceNet(nn.Module):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__()
        self.in_chs = 32
        self.num_feature = 1280
        self.reorg = Reorg(slice)
        self.conv_stem = conv3x3(3, self.in_chs, 2, slice=slice)
        self.bn1 = norm_layer(self.in_chs, slice=slice)
        self.act1 = act_layer()

        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi, slice=slice)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, slice=slice, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, slice=slice, exp_ratios=[4.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[3], out_channels[3], strides[3], ch_multi, slice=-slice))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[4], out_channels[4], strides[4], ch_multi, depth_multi, exp_ratios=[1.0, 6.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[5], out_channels[5], strides[5], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[6], out_channels[6], strides[6], ch_multi))
        self.blocks = nn.Sequential(*blocks)
        
        self.conv_head = conv1x1(self.in_chs, self.num_feature)
        self.bn2 = norm_layer(self.num_feature)
        self.act2 = act_layer()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_feature, num_class)

        self.weight_init()

    def forward(self, x):
        x = self.reorg(x)
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0], slice=1):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        num_block = int(math.ceil(num_block * depth_multiplier))
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        if slice < 0:
            slice = abs(slice)
            layers.append(block(self.in_chs, self.in_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            self.in_chs = self.in_chs * slice *slice
        else:
            layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice))
            self.in_chs = out_chs

        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i], slice=slice))
            self.in_chs = out_chs
        return nn.Sequential(*layers)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

def moslicenet():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 1, 1]
    num_blocks = [1, 2, 3, 1, 3, 3, 1]
    return MoSliceNet(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=4)
'''

