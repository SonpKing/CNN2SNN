import torch
import torch.nn as nn
import math
from .CommNet import get_exp_slice, norm_layer, make_divisible, round_channels, FuseSingle, Reorg
import numpy as np
import os
import pickle


def conv3x3(in_channels, out_channels, stride=1, slice=1, group=False):
    exp_slice = get_exp_slice(slice)
    if group:
        groups = out_channels * exp_slice
    else:
        groups = exp_slice
    return nn.Conv2d(in_channels * exp_slice, out_channels * exp_slice, kernel_size=3, stride=stride, padding=1, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, slice=1):
    exp_slice =  get_exp_slice(slice)
    return nn.Conv2d(in_channels * exp_slice, out_channels * exp_slice, kernel_size=1, stride=stride, groups=exp_slice)

from queue import PriorityQueue
class MyRelu(nn.Module):
    def __init__(self, total_num=50000, ratio=0.999):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.total_num = int(total_num * (1 - ratio))
        self.max_act = None

    def forward(self, x):
        x = self.relu(x)
        self.update_max_act(x)
        return x

    def update_max_act(self, x):
        if not self.max_act:
            max_size = np.array(x.shape[1:]).prod() * self.total_num
            print(max_size)
            self.max_act = PriorityQueue(maxsize=max_size)
        print("update", x.shape)
        tmp = float('-inf')
        for item in x.cpu().detach().numpy().ravel():
            if self.max_act.full():
                if item <= tmp:
                    continue
                small = self.max_act.get()
                tmp = max(item, small)
                item = tmp
            self.max_act.put(item)
        print("update end")

    def get_max_act(self):
        return self.max_act.queue
    

class MyRelu2(nn.Module):
    idx = 0
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.name = str(MyRelu2.idx)
        MyRelu2.idx += 1
        self.batch = 0

    def forward(self, x):
        x = self.relu(x)
        self.save_act(x)
        return x

    def save_act(self, x):
        new_dir = os.path.join("max_activations", self.name)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        self.batch += 1
        save_path = os.path.join(new_dir, str(self.batch))
        with open(save_path, "wb") as f:
            pickle.dump(x, f)
        



find_max_act = False
def act_layer():
    if find_max_act:
        return MyRelu2()
    else:
        return nn.ReLU(inplace=True)

class Scale(nn.Module):
    def __init__(self, scale=1):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([scale]), requires_grad=False)

    def forward(self, x):
        x *= self.scale  #pointwise multiple
        return x

    def set_scale(self, scale):
        self.scale.data *= scale

class DepthwiseConv(nn.Module):
    """ DepthwiseConv block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, stride=1, slice=1, **kwargs):
        super(DepthwiseConv, self).__init__()
        self.has_residual = stride == 1 and in_chs == out_chs

        self.conv_dw = conv3x3(in_chs, in_chs, stride, group=True, slice=slice)
        # self.bn1 = norm_layer(in_chs, slice=slice)
        self.act1 = act_layer()

        self.conv_pw = conv1x1(in_chs, out_chs, slice=slice)
        # self.bn2 = norm_layer(out_chs, slice=slice)
        self.act2 = act_layer()

        self.scale = Scale()

    def forward(self, x):
        residual = x
 
        x = self.conv_dw(x)
        # x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_pw(x)
        # x = self.bn2(x)

        if self.has_residual:
            x += self.scale(residual)

        x = self.act2(x)

        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, stride=1, exp_ratio=6.0, slice=1, real_in_chs = None):
        super(InvertedResidual, self).__init__()
        mid_chs = make_divisible(in_chs * exp_ratio)
        if real_in_chs:
            in_chs = real_in_chs
        self.has_residual = in_chs == out_chs and stride == 1

        # Point-wise expansion
        self.conv_pw = conv1x1(in_chs, mid_chs, slice=slice)
        # self.bn1 = norm_layer(mid_chs, slice=slice)
        self.act1 = act_layer()

        # Depth-wise convolution
        self.conv_dw = conv3x3(mid_chs, mid_chs, stride, group=True, slice=slice)
        # self.bn2 = norm_layer(mid_chs, slice=slice)
        self.act2 = act_layer()

        # Point-wise linear projection
        self.conv_pwl = conv1x1(mid_chs, out_chs, slice=slice)
        # self.bn3 = norm_layer(out_chs, slice=slice)
        self.act3 = act_layer()

        self.scale = Scale()

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        # x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        # x = self.bn2(x)
        x = self.act2(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        # x = self.bn3(x)

        if self.has_residual:
            x += self.scale(residual)

        x = self.act3(x)

        return x


class MoSliceNet(nn.Module):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0, slice = 1):
        super().__init__()
        self.in_chs = 32
        self.num_feature = 1280
        self.reorg = Reorg(slice)
        self.slice = slice
        self.conv_stem = conv3x3(3, self.in_chs, 2, slice=self.slice)
        # self.bn1 = norm_layer(self.in_chs, slice=self.slice)
        self.act1 = act_layer()

        
        self.blocks = self.make_blocks(num_blocks, out_channels, strides, ch_multi, depth_multi)
        
        self.conv_head = conv1x1(self.in_chs, self.num_feature, slice=self.slice)
        # self.bn2 = norm_layer(self.num_feature, self.slice)
        self.act2 = act_layer()
        self.global_pool = nn.AvgPool2d(4)
        self.act3 = act_layer()
        self.classifier = nn.Linear(self.num_feature, num_class)
        self.act4 = act_layer()
        self.weight_init()

    def forward(self, x):
        x = self.reorg(x)
        x = self.conv_stem(x)
        # x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        # x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = self.act3(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = self.act4(x)
        return x

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


def moslicenetv10_nobn(find_act=False):
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    global find_max_act
    find_max_act = find_act
    return MoSliceNet(1000, num_blocks, output_channels, strides, 1.2, 1.4, slice=2)



if __name__ == "__main__":
    net = moslicenetv10_nobn()
    # state = net.state_dict()
    # for layer in state:
    #     print(layer, state[layer].shape)
    for name, module in net.named_modules:
        print(name, module)
