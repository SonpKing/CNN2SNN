import torch
import torch.nn as nn
import pickle
import os
import numpy as np

class If(nn.Module):
    cnt = 0
    def __init__(self, V_th=1, V_reset=0):
        super().__init__()
        self.V_th = torch.tensor(V_th)
        self.membrane = None
        self.init = True
        self.total_num = None
        # self.idx = -1
        # self.bias = None

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
        if isinstance(m, If):
            m.reset()

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


class SpikeNet(nn.Module):
    def __init__(self, net, vth, scale=100):
        super().__init__()
        self.input = If2(scale)
        self.net = net
        self.output = If3(vth)
        self.scale = scale

    def forward(self, x):
        x = torch.round(x * self.scale)
        x = self.input(x)
        x = self.net(x)
        x = self.output(x)

        return x

    