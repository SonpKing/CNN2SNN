import torch
import torch.nn as nn
import math

def conv3x3(in_channels, out_channels, stride=1, group=False):
    if group:
        groups = out_channels
    else:
        groups = 1
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def norm_layer(out_channels):
    return nn.BatchNorm2d(out_channels)

def act_layer():
    return nn.ReLU(inplace=True)


class DepthwiseConv(nn.Module):
    """ DepthwiseConv block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, stride=1, **kwargs):
        super(DepthwiseConv, self).__init__()
        self.has_residual = stride == 1 and in_chs == out_chs

        self.conv_dw = conv3x3(in_chs, in_chs, stride, group=True)
        self.bn1 = norm_layer(in_chs)
        self.act1 = act_layer()

        self.conv_pw = conv1x1(in_chs, out_chs)
        self.bn2 = norm_layer(out_chs)
        # self.act2 = act_layer() if self.has_pw_act else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        # x = self.act2(x)

        if self.has_residual:
            x += residual
        return x

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)

class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, stride=1, exp_ratio=6.0):
        super(InvertedResidual, self).__init__()
        mid_chs = make_divisible(in_chs * exp_ratio)
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

        if self.has_residual:
            x += residual

        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0):
        super(MobileNetV2, self).__init__()
        self.in_chs = 32
        self.num_feature = 1280
        self.conv_stem = conv3x3(3, self.in_chs, 2)
        self.bn1 = nn.BatchNorm2d(self.in_chs)
        self.act1 = act_layer()

        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi)]
        for i in range(1, len(num_blocks)-1):
            blocks.append(self.make_layers(InvertedResidual, num_blocks[i], out_channels[i], strides[i], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[-1], out_channels[-1], strides[-1], ch_multi))
        self.blocks = nn.Sequential(*blocks)
        
        self.conv_head = conv1x1(self.in_chs, self.num_feature)
        self.bn2 = norm_layer(self.num_feature)
        self.act2 = act_layer()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_feature, num_class)

        self.weight_init()

    def forward(self, x):
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

    def make_layers(self, block, num_block, out_chs, stride, channel_multiplier=1.0, depth_multiplier=1.0, exp_ratios=[6.0]):
        layers = []
        out_chs = round_channels(out_chs, multiplier=channel_multiplier)
        layers.append(block(self.in_chs, out_chs, stride, exp_ratio=exp_ratios[0]))
        num_block = int(math.ceil(num_block * depth_multiplier))
        self.in_chs = out_chs
        if len(exp_ratios) < num_block:
            exp_ratios = exp_ratios + [exp_ratios[-1]] * (num_block - len(exp_ratios))
        for i in range(1, num_block):
            layers.append(block(self.in_chs, out_chs, exp_ratio=exp_ratios[i]))
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


def mobilenetv2_120():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MobileNetV2(1000, num_blocks, output_channels, strides, 1.2, 1.4)

class MobileDarwinNet(MobileNetV2):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0):
        super(MobileDarwinNet, self).__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi)
        self.in_chs = 32

        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, exp_ratios=[4.0, 6.0]))
        for i in range(3, len(num_blocks)-1):
            blocks.append(self.make_layers(InvertedResidual, num_blocks[i], out_channels[i], strides[i], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[-1], out_channels[-1], strides[-1], ch_multi))
        self.blocks = nn.Sequential(*blocks)

        self.weight_init()

def mobile_darwinnet():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MobileDarwinNet(1000, num_blocks, output_channels, strides, 1.2, 1.4)

def mobile_darwinnetv2():
    output_channels = [16, 24, 32, 64, 96, 160]
    strides = [1, 2, 2, 2, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 1]
    return MobileDarwinNet(1000, num_blocks, output_channels, strides, 1.2, 1.4)

def mobile_darwinnetv3():
    output_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 2, 2, 2, 1, 1, 1]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    return MobileDarwinNet(1000, num_blocks, output_channels, strides, 1.2, 1.4)

class MobileDarwinNetV2(MobileNetV2):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi)
        self.in_chs = 32
        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[4.0]))
        for i in range(2, len(num_blocks)-1):
            blocks.append(self.make_layers(InvertedResidual, num_blocks[i], out_channels[i], strides[i], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[-1], out_channels[-1], strides[-1], ch_multi))
        self.blocks = nn.Sequential(*blocks)

        self.weight_init()



class MobileDarwinNetV4(MobileNetV2):
    def __init__(self, num_class, num_blocks, out_channels, strides, ch_multi=1.0, depth_multi=1.0):
        super().__init__(num_class, num_blocks, out_channels, strides, ch_multi=ch_multi, depth_multi=depth_multi)
        self.in_chs = 32

        blocks = [self.make_layers(DepthwiseConv, num_blocks[0], out_channels[0], strides[0], ch_multi)]
        blocks.append(self.make_layers(InvertedResidual, num_blocks[1], out_channels[1], strides[1], ch_multi, depth_multi, exp_ratios=[2.0, 4.0]))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[2], out_channels[2], strides[2], ch_multi, depth_multi, exp_ratios=[4.0, 6.0]))
        for i in range(3, len(num_blocks)-1):
            blocks.append(self.make_layers(InvertedResidual, num_blocks[i], out_channels[i], strides[i], ch_multi, depth_multi))
        blocks.append(self.make_layers(InvertedResidual, num_blocks[-1], out_channels[-1], strides[-1], ch_multi))
        self.blocks = nn.Sequential(*blocks)

        self.weight_init()

def mobile_darwinnetv4():
    output_channels = [24, 24, 32, 64, 96, 160]
    strides = [1, 2, 2, 2, 1, 1]
    num_blocks = [3, 4, 4, 2, 2, 1]
    return MobileDarwinNetV4(1000, num_blocks, output_channels, strides, 1.2, 1.4) 



if __name__ == "__main__":
    # print()
    # print(mobile_darwinnet())
    # print(mobile_darwinnetv2())
    # from torchsummary import summary
    # import torch.backends.cudnn as cudnn
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="2"

    # model =  mobile_darwinnetv2().cuda()
    # cudnn.benchmark = True
    # summary(model, (3, 64, 64))
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from torch.autograd import Variable
    dummy_input = Variable(torch.rand(1, 3, 64, 64)) #假设输入13张1*28*28的图片
    model = mobile_darwinnetv4()
    with SummaryWriter(comment='mobile_darwinnet') as w:
        w.add_graph(model, (dummy_input, ))
        w.close()
    # # %%
    # import torch
    # from torchviz import make_dot
    # from MobileNet import mobile_darwinnetv2
    # model = mobile_darwinnetv2()
    # dummy_input = torch.autograd.Variable(torch.rand(1, 3, 64, 64))
    # vis_graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    # vis_graph.view()
    
    

# %%
