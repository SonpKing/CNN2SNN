import torch
import torch.nn as nn
import math
import pickle
import os
from .SpikeNet import *

def conv3x3(in_channels, out_channels, stride=1, group=False, bias=False):
    if group:
        groups = out_channels
    else:
        groups = 1
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

def norm_layer(out_channels):
    return nn.BatchNorm2d(out_channels)

class Pool_Scale(nn.Module):
    def __init__(self, scale=1):
        super(Pool_Scale, self).__init__()
        self.pool = nn.AvgPool2d(4)
        self.scale = Scale()

    def forward(self, x):
        x = self.pool(x)
        x = self.scale(x)
        return x


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


class DepthwiseConv2(nn.Module):
    """ DepthwiseConv block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, stride=1, **kwargs):
        super().__init__()
        self.has_residual = stride == 1 and in_chs == out_chs

        self.conv_dw = conv3x3(in_chs, in_chs, stride, group=True)
        self.bn1 = norm_layer(in_chs)
        self.act1 = act_layer()

        self.conv_pw = conv1x1(in_chs, out_chs)
        self.bn2 = norm_layer(out_chs)
        self.act2 = act_layer()

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_pw(x)
        x = self.bn2(x)

        # if self.has_residual:
        #     x += residual

        x = self.act2(x)
        return x


class InvertedResidual2(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, stride=1, exp_ratio=6.0):
        super().__init__()
        mid_chs = in_chs * exp_ratio
        self.has_residual = in_chs == out_chs and stride == 1

        # Point-wise expansion
        self.conv_pw = conv1x1(in_chs, mid_chs)
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer()

        # Depth-wise convolution
        self.conv_dw = conv3x3(mid_chs, mid_chs, stride, group=True)
        self.bn2 = norm_layer(mid_chs)
        self.act2 = act_layer()

        # Point-wise linear projection
        self.conv_pwl = conv1x1(mid_chs, out_chs)
        self.bn3 = norm_layer(out_chs)
        self.act3 = act_layer()

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        # if self.has_residual:
        #     x += residual

        x = self.act3(x)
        return x


BIAS = True
class DepthwiseConv(nn.Module):
    """ DepthwiseConv block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, stride=1, **kwargs):
        super().__init__()
        self.has_residual = False
        self.conv_dw = conv3x3(in_chs, in_chs, stride, group=True, bias=BIAS)
        self.act1 = act_layer()

        self.conv_pw = conv1x1(in_chs, out_chs, bias=BIAS)
        self.act2 = act_layer() 

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.act1(x)
        x = self.conv_pw(x)
        x = self.act2(x)
        return x

class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, stride=1, exp_ratio=6.0):
        super().__init__()
        self.has_residual = False
        mid_chs = in_chs * exp_ratio

        # Point-wise expansion
        self.conv_pw = conv1x1(in_chs, mid_chs, bias=BIAS)
        self.act1 = act_layer()

        # Depth-wise convolution
        self.conv_dw = conv3x3(mid_chs, mid_chs, stride, group=True,  bias=BIAS)
        self.act2 = act_layer()

        # Point-wise linear projection
        self.conv_pwl = conv1x1(mid_chs, out_chs,  bias=BIAS)
        self.act3 = act_layer()

    def forward(self, x):
        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.act2(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.act3(x)

        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_class, num_blocks, out_channels, strides):
        super(MobileNetV2, self).__init__()
        self.in_chs = 32
        self.num_feature = 256
        self.conv_stem = conv3x3(3, self.in_chs, 2)
        self.act1 = act_layer()

        self.blocks = self.make_blocks(num_blocks, out_channels, strides)

        self.conv_head = conv1x1(self.in_chs, self.num_feature)
        self.act2 = act_layer()
        self.global_pool = Pool_Scale()
        self.act3 = act_layer()
        # self.drop = torch.nn.Dropout(0.5)
        self.classifier = nn.Linear(self.num_feature, num_class, bias=False)
        if SAVE_ACT:
            self.act4 = act_layer()  #need to find max_activation
        self.weight_init()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.act1(x)
        # print("conv_stem", x.cpu().detach().numpy().ravel()[:10])
        x = self.blocks(x)
        # print("blocks", x.cpu().detach().numpy().ravel()[:10])
        x = self.conv_head(x)
        x = self.act2(x)
        # print("conv_head", x.cpu().detach().numpy().ravel()[:10])
        x = self.global_pool(x)
        
        x = self.act3(x)
        # x = self.drop(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        if SAVE_ACT:
            x = self.act4(x)
        return x

    def make_blocks(self, num_blocks, out_channels, strides):
        return nn.Identity()

    def make_layers(self, block, num_block, out_chs, stride, exp_ratios=[6.0]):
        layers = []
        layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0]))
        self.in_chs = out_chs
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i]))
        return nn.Sequential(*layers)

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




class MobileNetV2_1(MobileNetV2):
    def __init__(self, num_class, num_blocks, out_channels, strides):
        super().__init__(num_class, num_blocks, out_channels, strides)

    def make_blocks(self, num_blocks, out_channels, strides):
        blocks = [self.make_layers(DepthwiseConv2, num_blocks[0], out_channels[0], strides[0])]
        blocks.append(self.make_layers(InvertedResidual2, num_blocks[1], out_channels[1], strides[1], exp_ratios=[2, 4]))
        blocks.append(self.make_layers(InvertedResidual2, num_blocks[2], out_channels[2], strides[2], exp_ratios=[4, 6]))
        return nn.Sequential(*blocks)


def mobilenet_slim_v1(class_num=100):
    output_channels = [24, 32, 40]
    strides = [1, 2, 2]
    num_blocks = [1, 2, 3]
    return MobileNetV2(class_num, num_blocks, output_channels, strides)


class MobileNetV2_2(MobileNetV2):
    def __init__(self, num_class, num_blocks, out_channels, strides):
        super().__init__(num_class, num_blocks, out_channels, strides)

    def make_blocks(self, num_blocks, out_channels, strides):
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0])]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], exp_ratios=[2, 4]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], exp_ratios=[4, 6]))
        return nn.Sequential(*blocks)


def mobilenet_slim_2(class_num=100, bias=True):
    global BIAS
    BIAS = bias
    output_channels = [24, 32, 40]
    strides = [1, 2, 2]
    num_blocks = [1, 2, 3]
    return MobileNetV2_2(class_num, num_blocks, output_channels, strides)



def mobilenet_slim_spike(class_num=100, bias=False, _if=True, save_act=False, vth=1.0):
    global BIAS
    BIAS = bias
    global IF
    IF = _if
    global SAVE_ACT
    SAVE_ACT = save_act
    global SCALE
    SCALE = vth
    output_channels = [24, 32, 40]
    strides = [1, 2, 2]
    num_blocks = [1, 2, 3]
    return MobileNetV2_2(class_num, num_blocks, output_channels, strides)


if __name__ == "__main__":
    from torchsummary import summary
    import torch.backends.cudnn as cudnn
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    cudnn.benchmark = True
    net = mobilenet_slim_spike(8).cuda()
    # for _ in range(10):
    #     input = torch.randn((1, 3, 128, 128)).cuda()
    #     print(net(input))
    summary(net, (3, 32, 32))