import torch
import torch.nn as nn
import math
import pickle
import os
from .SpikeNet import *

def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

def norm_layer(out_channels):
    return nn.BatchNorm2d(out_channels)


SAVE_ACT = False
IF = False
SCALE = 1.0
def act_layer():
    if IF:
        return If2(SCALE)
    elif SAVE_ACT:
        return MyRelu2()
    else:
        return nn.ReLU(inplace=True)

BIAS = True

class Pool_Scale(nn.Module):
    def __init__(self, scale=1):
        super(Pool_Scale, self).__init__()
        self.pool = nn.AvgPool2d(2) #nn.AdaptiveAvgPool2d((1, 1))
        self.scale = Scale()

    def forward(self, x):
        x = self.pool(x)
        x = self.scale(x)
        return x

class Blocks(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""
    def __init__(self, in_chs, out_chs, stride=1):
        super().__init__()
        self.conv= conv3x3(in_chs, out_chs, stride, bias=False)
        self.bn = norm_layer(out_chs)
        self.act = act_layer()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)

        return x

class Blocks2(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""
    def __init__(self, in_chs, out_chs, stride=1):
        super().__init__()
        self.conv= conv3x3(in_chs, out_chs, stride, bias=BIAS)
        self.act = act_layer()
        self.pool = Pool_Scale()
        if IF:
            self.act2 = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        if IF:
            x = self.act2(x)

        return x

class BananaNet(nn.Module):
    def __init__(self, num_class, out_channels, strides):
        super().__init__()
        self.in_chs = 3
        self.num_feature1 = 4*4*out_channels[-1]
        self.num_feature = 100

        self.blocks = self.make_blocks(out_channels, strides)

        self.fc1 = nn.Linear(self.num_feature1, self.num_feature, bias=False)
        self.act1 = act_layer()

        self.fc2 = nn.Linear(self.num_feature, num_class, bias=False)
        if SAVE_ACT:
            self.act2 = act_layer()

        self.weight_init()

    def make_blocks(self, out_channels, strides):
        return nn.Identity()

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        if SAVE_ACT:
            x = self.act2(x)

        return x

    def weight_init(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

class BananaNetV1(BananaNet):
    def __init__(self, num_class, out_channels, strides):
        super().__init__(num_class, out_channels, strides)

    def make_blocks(self, out_channels, strides):
        layers = []
        for (ch, stride) in zip(out_channels, strides):
            layers.append(Blocks(self.in_chs, ch, stride))
            self.in_chs = ch
        return nn.Sequential(*layers)

def banananetv1(class_num):
    out_channels = [14, 22, 28]
    strides = [1, 1, 1]
    return BananaNetV1(class_num, out_channels, strides)

class BananaNetV2(BananaNet):
    def __init__(self, num_class, out_channels, strides):
        super().__init__(num_class, out_channels, strides)

    def make_blocks(self, out_channels, strides):
        layers = []
        for (ch, stride) in zip(out_channels, strides):
            layers.append(Blocks2(self.in_chs, ch, stride))
            self.in_chs = ch
        return nn.Sequential(*layers)

def banananetv2(class_num, bias=False, if_=False, save_act=False):
    out_channels = [14, 22, 28]
    strides = [1, 1, 1]
    global BIAS
    BIAS = bias
    global IF
    IF = if_
    global SAVE_ACT
    SAVE_ACT = save_act
    return BananaNetV2(class_num, out_channels, strides)

def banananetv2_spike(class_num, vth):
    out_channels = [14, 22, 28]
    strides = [1, 1, 1]
    global BIAS
    BIAS = False
    global IF
    IF = True
    global SCALE
    SCALE = vth
    return BananaNetV2(class_num, out_channels, strides)

if __name__ == "__main__":
    from torchsummary import summary
    import torch.backends.cudnn as cudnn
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    cudnn.benchmark = True
    net = banananetv1(7).cuda()
    # for _ in range(10):
    #     input = torch.randn((1, 3, 128, 128)).cuda()
    #     print(net(input))
    summary(net, (3, 32, 32))
    print(net)
