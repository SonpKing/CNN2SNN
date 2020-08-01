import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

is_save_activations = False
def set_save_activations():
    global is_save_activations
    is_save_activations = True
activations = []


'''Todo
sparity
dilated 
'''

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, self.expansion * channels)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual= x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if is_save_activations: 
            activations.append(out.cpu().data.numpy())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if is_save_activations: 
            activations.append(out.cpu().data.numpy())
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        if is_save_activations: 
            activations.append(out.cpu().data.numpy())

        return out

class ResNet(nn.Module):
    def __init__(self, bottlenect, layers, channels=16, num_classes=100):
        self.inplanes = channels
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.AvgPool2d((3, 3), 2)
        self.layer1 = self._make_layer(bottlenect, channels, layers[0], stride=2)
        self.layer2 = self._make_layer(bottlenect, channels*2, layers[1], stride=2)
        self.layer3 = self._make_layer(bottlenect, channels*4, layers[2], stride=2)
        self.layer4 = self._make_layer(bottlenect, channels*8, layers[3], stride=1, dilation=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc =  nn.Linear(channels*8 * bottlenect.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d): 
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if is_save_activations: 
            activations.clear()
            activations.append(out.cpu().data.numpy())


        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        if is_save_activations: 
            activations.append(out.cpu().data.numpy())

        return out

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:  #downsample for residual link
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, 
            kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes*block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes*block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

def resnet35(class_num, init_channels=16, pretrained=False, pretrained_path=None):
    model = ResNet(Bottleneck, [2,4,3,2], channels=init_channels, num_classes=class_num) # 2+4+4+1=11, 33*3+2=35
    if pretrained and pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
    return model


class ResNetYoLo(ResNet):
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out1 = self.layer3(out)
        out2 = self.layer4(out1)
        return out1, out2

def resnet_yolo(init_channels=16, pretrained=False, pretrained_path=None):
    model = ResNetYoLo(Bottleneck, [2,4,3,2], channels=init_channels, num_classes=1000) # 2+4+4+1=11, 33*3+2=35
    if pretrained and pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
    return model


if __name__ == "__main__":
# %%
    from torchsummary import summary
    import torch.backends.cudnn as cudnn
    from torch.autograd import Variable
    from resnet import resnet35
    model = resnet35(1000)
    print(model)
    # cudnn.benchmark = True
    # summary(model, (3, 80, 80))
    # dummy_input = Variable(torch.rand(1, 3, 64, 64))
    # from torch.utils.tensorboard import SummaryWriter
    # with SummaryWriter(comment='resnet') as w:
    #     w.add_graph(model, (dummy_input, ))



# %%
