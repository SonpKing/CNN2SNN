import torch
import torch.nn as nn
import math
import pickle
import os

def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

def norm_layer(out_channels):
    return nn.BatchNorm2d(out_channels)


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
        print("save activations", self.name)
        new_dir = os.path.join("max_activations", self.name)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        self.batch += 1
        save_path = os.path.join(new_dir, str(self.batch))
        with open(save_path, "wb") as f:
            pickle.dump(x, f)

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

class Scale(nn.Module):
    def __init__(self, scale=1):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.Tensor([scale]), requires_grad=False)
        self.bias = None

    def forward(self, x):
        x *= self.weight  #pointwise multiple
        return x

    def set_scale(self, scale):
        self.weight.data *= scale

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

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class Blocks2(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""
    def __init__(self, in_chs, out_chs, stride=1):
        super().__init__()
        self.conv= conv3x3(in_chs, out_chs, stride, bias=BIAS)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x

class AppleNet(nn.Module):
    def __init__(self, num_class, out_channels, strides):
        super().__init__()
        self.in_chs = 3
        self.num_feature = 128

        self.blocks = self.make_blocks(out_channels, strides)

        self.pool = Pool_Scale()
        self.act1 = act_layer()

        self.fc = nn.Linear(self.num_feature, num_class, bias=False)
        self.act2 = act_layer()
        
        self.weight_init()

    def make_blocks(self, out_channels, strides):
        return nn.Identity()

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        x = self.act1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
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

class AppleNetV1(AppleNet):
    def __init__(self, num_class, out_channels, strides):
        super().__init__(num_class, out_channels, strides)

    def make_blocks(self, out_channels, strides):
        layers = []
        for (ch, stride) in zip(out_channels, strides):
            layers.append(Blocks(self.in_chs, ch, stride))
            self.in_chs = ch
        return nn.Sequential(*layers)

def applenetv1(class_num):
    out_channels = [32, 32, 64, 128]
    strides = [2, 2, 2, 2]
    return AppleNetV1(class_num, out_channels, strides)

class AppleNetV2(AppleNet):
    def __init__(self, num_class, out_channels, strides):
        super().__init__(num_class, out_channels, strides)

    def make_blocks(self, out_channels, strides):
        layers = []
        for (ch, stride) in zip(out_channels, strides):
            layers.append(Blocks2(self.in_chs, ch, stride))
            self.in_chs = ch
        return nn.Sequential(*layers)

def applenetv2(class_num, bias=False, if_=False, save_act=False):
    out_channels = [32, 32, 64, 128]
    strides = [2, 2, 2, 2]
    global BIAS
    BIAS = bias
    global IF
    IF = if_
    global SAVE_ACT
    SAVE_ACT = save_act
    return AppleNetV2(class_num, out_channels, strides)

def applenetv2_spike(class_num, vth):
    out_channels = [32, 32, 64, 128]
    strides = [2, 2, 2, 2]
    global BIAS
    BIAS = False
    global IF
    IF = True
    global SCALE
    SCALE = vth
    return AppleNetV2(class_num, out_channels, strides)


class If(nn.Module):
    cnt = 0
    def __init__(self, V_th=1, V_reset=0):
        super().__init__()
        self.V_th = torch.tensor(V_th)
        self.V_reset = V_reset
        self.membrane = None
        self.init = True
        self.total_num = None
        self.idx = -1
        self.bias = None

    def create_neurons(self, x):
        self.membrane = torch.zeros(x.shape, device = x.device) #！！！maybe increase gpu memory
        self.total_num = torch.zeros(x.shape, device = x.device)
        If.cnt += torch.numel(self.membrane)

    def forward(self, x):
        if self.init or self.membrane.shape!=x.shape:
            self.create_neurons(x)
            self.init=False
        self.membrane += x
        spike = self.membrane >= self.V_th
        spike =  self.membrane * spike.int() // self.V_th
        self.membrane -= spike * self.V_th
        self.total_num += spike
        return spike.float()

    def reset(self):
        if not self.init:
            self.membrane *= 0.0  
            self.total_num *= 0

class If2(If):
    def forward(self, x):
        if self.init or self.membrane.shape!=x.shape:
            self.create_neurons(x)
            self.init=False
        self.membrane += x
        spike = (self.membrane >= self.V_th).int()
        self.membrane -= spike * self.V_th
        self.total_num += spike
        return spike.float()


class If3(If):
    def forward(self, x):
        if self.init or self.membrane.shape!=x.shape:
            self.create_neurons(x)
            self.init=False
        self.membrane += x
        spike = self.membrane >= self.V_th
        self.membrane[spike] = 0
        self.total_num += spike.int()
        return spike.float()


def reset_spikenet(net):
    print("reset spikenet")
    for _, m in net.named_modules():
        if isinstance(m, If) or isinstance(m, If2):
            m.reset()

if __name__ == "__main__":
    from torchsummary import summary
    import torch.backends.cudnn as cudnn
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    cudnn.benchmark = True
    net = applenetv2(8).cuda()
    # for _ in range(10):
    #     input = torch.randn((1, 3, 128, 128)).cuda()
    #     print(net(input))
    summary(net, (3, 32, 32))
    print(net)
