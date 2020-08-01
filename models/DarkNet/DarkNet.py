import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import cv2
import os, sys

class Conv_Bn_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=-1, stride=1):
        super(Conv_Bn_Relu, self).__init__()
        if padding==-1: 
            padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    '''repeat network layer'''
    def __init__(self, net_layer, n_layers, in_channels):
        super(Bottleneck, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers//2):
            self.layers.append(net_layer(in_channels, 2*in_channels, 3))
            self.layers.append(net_layer(2*in_channels, in_channels, 1))
        self.layers.append(net_layer(in_channels, 2*in_channels, 3))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        


class DarkNet(nn.Module):
    '''Like ResNet, this network is used for feature extraction
    step 1, we train this net to classify the images
    setp 2, we modify the net to genenrate the bounding box
    '''
    def __init__(self, init_channels=32, classify = True, n_class=1000):
        super(DarkNet, self).__init__()
        channels = init_channels
        self.classify = classify
        self.conv1 = Conv_Bn_Relu(3, channels, 3)
        self.pool1 = nn.AvgPool2d((2, 2), 2)

        self.conv2 = Conv_Bn_Relu(channels, 2*channels, 3)
        self.pool2 = nn.AvgPool2d((2, 2), 2)

        self.conv3 = Bottleneck(Conv_Bn_Relu, 3, 2*channels)
        self.pool3 = nn.AvgPool2d((2, 2), 2)

        self.conv4 = Bottleneck(Conv_Bn_Relu, 3, 4*channels)
        self.pool4 = nn.AvgPool2d((2, 2), 2)

        self.conv5 = Bottleneck(Conv_Bn_Relu, 5, 8*channels)
        self.pool5 = nn.AvgPool2d((2, 2), 2)

        self.conv6 = Bottleneck(Conv_Bn_Relu, 5, 16*channels)

        if self.classify:
            self.pool6 = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(32*channels, n_class) 


    def forward(self, x):
        out = self.pool1(self.conv1(x))
        out = self.pool2(self.conv2(out))
        out = self.pool3(self.conv3(out))
        out = self.pool4(self.conv4(out))
        out2 = self.conv5(out)
        out = self.conv6(self.pool5(out2))
        if self.classify:
            out = self.pool6(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
        else:
            return out, out2






        

