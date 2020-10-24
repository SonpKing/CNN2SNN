import torch
import torch.nn as nn

def get_exp_slice(slice):
    '''
    get the number of expand slice.
    ------------------------------
    if slice is a int number, the we split the layer into slice * slice smaller layers
    if slice is a int list, the total number is the sum of each slice^2 in the list
    '''
    if isinstance(slice, int):
        exp_slice =  slice * slice
    elif isinstance(slice, list):
        exp_slice = 0
        for ch_slice in slice:
            exp_slice += ch_slice * ch_slice
    else:
        assert "slice type error"==""
    return exp_slice

def conv3x3(in_channels, out_channels, stride=1, slice=1, group=False):
    '''
    get convolution layer whose kernel size is 3
    Parameters
    ----------
    in_channels: int
        the number of input layer's channels, 
    out_channels: int
        the number of output layer's channels
    stride: int
        stride size of the layer
    slice: int or list
        use to compute numbers of split smaller layers. multi by groups
    group: bool
        if true, then init groups is out_channels. else init groups is 1
    Notes
    -----
    default slice is 1
    '''
    exp_slice = get_exp_slice(slice)
    if group:
        groups = out_channels * exp_slice
    else:
        groups = exp_slice
    return nn.Conv2d(in_channels * exp_slice, out_channels * exp_slice, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, slice=1):
    exp_slice =  get_exp_slice(slice)
    return nn.Conv2d(in_channels * exp_slice, out_channels * exp_slice, kernel_size=1, stride=stride, bias=False, groups=exp_slice)

def norm_layer(out_channels, slice=1):
    exp_slice =  get_exp_slice(slice)
    return nn.BatchNorm2d(out_channels * exp_slice)

def act_layer():
    return nn.ReLU(inplace=True)


class DepthwiseConv(nn.Module):
    """ DepthwiseConv block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, stride=1, slice=1, **kwargs):
        super(DepthwiseConv, self).__init__()
        self.has_residual = stride == 1 and in_chs == out_chs

        self.conv_dw = conv3x3(in_chs, in_chs, stride, group=True, slice=slice)
        self.bn1 = norm_layer(in_chs, slice=slice)
        self.act1 = act_layer()

        self.conv_pw = conv1x1(in_chs, out_chs, slice=slice)
        self.bn2 = norm_layer(out_chs, slice=slice)
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

class DepthwiseConvRelu(nn.Module):
    """ DepthwiseConv block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, stride=1, slice=1, **kwargs):
        super(DepthwiseConvRelu, self).__init__()
        self.has_residual = stride == 1 and in_chs == out_chs

        self.conv_dw = conv3x3(in_chs, in_chs, stride, group=True, slice=slice)
        self.bn1 = norm_layer(in_chs, slice=slice)
        self.act1 = act_layer()

        self.conv_pw = conv1x1(in_chs, out_chs, slice=slice)
        self.bn2 = norm_layer(out_chs, slice=slice)
        self.act2 = act_layer()

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_pw(x)
        x = self.bn2(x)

        if self.has_residual:
            x += residual

        x = self.act2(x)
        return x

class DepthwiseConvNoRes(nn.Module):
    """ DepthwiseConv block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, stride=1, slice=1, **kwargs):
        super(DepthwiseConvNoRes, self).__init__()
        self.has_residual = stride == 1 and in_chs == out_chs

        self.conv_dw = conv3x3(in_chs, in_chs, stride, group=True, slice=slice)
        self.bn1 = norm_layer(in_chs, slice=slice)
        self.act1 = act_layer()

        self.conv_pw = conv1x1(in_chs, out_chs, slice=slice)
        self.bn2 = norm_layer(out_chs, slice=slice)
        # self.act2 = act_layer() if self.has_pw_act else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        # x = self.act2(x)

        # if self.has_residual:
        #     x += residual
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

class InvertedResidualRelu(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, stride=1, exp_ratio=6.0, slice=1, real_in_chs = None):
        super(InvertedResidualRelu, self).__init__()
        mid_chs = make_divisible(in_chs * exp_ratio)
        if real_in_chs:
            in_chs = real_in_chs
        self.has_residual = in_chs == out_chs and stride == 1

        # Point-wise expansion
        self.conv_pw = conv1x1(in_chs, mid_chs, slice=slice)
        self.bn1 = norm_layer(mid_chs, slice=slice)
        self.act1 = act_layer()

        # Depth-wise convolution
        self.conv_dw = conv3x3(mid_chs, mid_chs, stride, group=True, slice=slice)
        self.bn2 = norm_layer(mid_chs, slice=slice)
        self.act2 = act_layer()

        # Point-wise linear projection
        self.conv_pwl = conv1x1(mid_chs, out_chs, slice=slice)
        self.bn3 = norm_layer(out_chs, slice=slice)
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

        if self.has_residual:
            x += residual

        x = self.act3(x)
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
        self.bn1 = norm_layer(mid_chs, slice=slice)
        self.act1 = act_layer()

        # Depth-wise convolution
        self.conv_dw = conv3x3(mid_chs, mid_chs, stride, group=True, slice=slice)
        self.bn2 = norm_layer(mid_chs, slice=slice)
        self.act2 = act_layer()

        # Point-wise linear projection
        self.conv_pwl = conv1x1(mid_chs, out_chs, slice=slice)
        self.bn3 = norm_layer(out_chs, slice=slice)

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

class InvertedResidualNoRes(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, stride=1, exp_ratio=6.0, slice=1, real_in_chs = None):
        super(InvertedResidualNoRes, self).__init__()
        mid_chs = make_divisible(in_chs * exp_ratio)
        if real_in_chs:
            in_chs = real_in_chs
        self.has_residual = stride == 1 and in_chs == out_chs

        # Point-wise expansion
        self.conv_pw = conv1x1(in_chs, mid_chs, slice=slice)
        self.bn1 = norm_layer(mid_chs, slice=slice)
        self.act1 = act_layer()

        # Depth-wise convolution
        self.conv_dw = conv3x3(mid_chs, mid_chs, stride, group=True, slice=slice)
        self.bn2 = norm_layer(mid_chs, slice=slice)
        self.act2 = act_layer()

        # Point-wise linear projection
        self.conv_pwl = conv1x1(mid_chs, out_chs, slice=slice)
        self.bn3 = norm_layer(out_chs, slice=slice)

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

        return x

class Reorg(nn.Module):
    def __init__(self, slice):
        super().__init__()
        if isinstance(slice, int):
            self.reorg = ReorgSw(slice=slice)
        elif isinstance(slice, list):
            self.reorg = ReorgMix(slice=slice)
        else:
            exit(-1)
        
    def forward(self, x):
        return self.reorg(x)

class ReorgMix(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.reorg1 = ReorgSw(slice=slice[0])
        self.reorg2 = ReorgPw(stride=slice[1])

    def forward(self, x):
        x1 = self.reorg1(x)
        x2 = self.reorg2(x)
        return torch.cat((x1, x2), 1)

class ReorgSw(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.slice = slice

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        _height, _width = height // self.slice, width // self.slice
        x = x.view(batch_size, channels, self.slice, _height, self.slice, _width).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, self.slice * self.slice, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, channels * self.slice * self.slice, _height, _width)
        return x

class ReorgPw(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        _height, _width = height // self.stride, width // self.stride

        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x


class ReorgPyramid(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg2 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.reorg = ReorgSw(slice=slice[0])
        self.reorg1 = ReorgSw(slice=slice[1])

    def forward(self, x):
        x1 = self.reorg(x)
        x2 = self.avg1(x)
        x2 = self.reorg1(x2)
        x3 = self.avg2(x)
        return torch.cat((x1, x2, x3), 1)

class ReorgPyramid2(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.reorg = ReorgSw(slice=slice[0])


    def forward(self, x):
        x1 = self.reorg(x)
        x2 = self.avg1(x)
        return torch.cat((x1, x2), 1)


def generate_pos(total_len, slice, overlap):
    part_len = int(total_len / slice)
    overlap_len = int((part_len * int(slice + overlap) - total_len) // int(slice + overlap - 1))
    cur_len = 0
    res = []
    while cur_len + part_len <= total_len:
        res.append(cur_len)
        cur_len += part_len - overlap_len
    return torch.tensor(res)


class ReorgOverlap(nn.Module):
    def __init__(self, slice, overlap=1):
        super().__init__()
        self.slice = slice
        self.overlap = overlap
        self.has_grid = False

    def forward(self, x):
        if not self.has_grid:
            batch_size, channels, height, width = x.shape
            self.generate_grid(height, width)
        patches = []
        for i in range(len(self.hpos)):
            for j in range(len(self.wpos)):
                patches.append(x[:, :, self.hpos[i]: self.hpos[i] + self.hpatch, self.wpos[j]: self.wpos[j] + self.wpatch])
        return torch.cat(patches, 1)
        
    def generate_grid(self, height, width):
        self.hpos = generate_pos(height, self.slice, self.overlap)
        self.wpos = generate_pos(width, self.slice, self.overlap)
        self.hpatch = int(height / self.slice)
        self.wpatch = int(width / self.slice)
        self.has_grid = True
        
        

def generate_pos2(total_len, part_len, overlap_len):
    cur_len = 0
    res = []
    while cur_len + part_len <= total_len:
        res.append(cur_len)
        cur_len += part_len - overlap_len
    return torch.tensor(res)


class ReorgOverlap2(nn.Module):
    def __init__(self, patch, overlap):
        super().__init__()
        self.patch = patch
        self.overlap = overlap
        self.has_grid = False

    def forward(self, x):
        if not self.has_grid:
            batch_size, channels, height, width = x.shape
            self.generate_grid(height, width)
        patches = []
        for i in range(len(self.hpos)):
            for j in range(len(self.wpos)):
                patches.append(x[:, :, self.hpos[i]: self.hpos[i] + self.hpatch, self.wpos[j]: self.wpos[j] + self.wpatch])
        return torch.cat(patches, 1)
        
    def generate_grid(self, height, width):
        self.hpos = generate_pos2(height, self.patch, self.overlap)
        self.wpos = generate_pos2(width, self.patch, self.overlap)
        self.hpatch = self.patch
        self.wpatch = self.patch
        self.has_grid = True
        # print(self.hpos, self.wpos)

class InvertedReorg(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.slice = slice

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        _height, _width = height * self.slice, width * self.slice
        channels = channels // self.slice // self.slice
        x = x.view(batch_size, self.slice * self.slice, channels, height, width).transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, self.slice, self.slice, height, width).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height, _width)
        
        return x

class Reshuffle(nn.Module):
    def __init__(self, img_size=8, patch=2, stride=2):
        super().__init__()
        loop = img_size // patch // stride
        self.shuffle = []
        for i in range(stride):
            for j in range(loop):  
                self.shuffle += list(range(i*patch+ j*patch*stride, i*patch+patch + j*patch*stride))
        # print(self.shuffle)

    def forward(self, x):
        y = x[:, :, self.shuffle, :]
        y = y[:, :, :, self.shuffle]
        return y

class FuseSingle(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.invReorg = InvertedReorg(slice = slice)
        self.reorg = Reorg(slice = slice//2)


    def forward(self, x):
        x = self.invReorg(x)
        x = self.reorg(x)

        return x

class Fuse(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.reshuffle = Reshuffle()
        self.invReorg = InvertedReorg(slice = slice)
        self.reorg = Reorg(slice = slice//2)


    def forward(self, x):
        x = self.invReorg(x)
        y = self.reshuffle(x)
        x = self.reorg(x)
        y = self.reorg(y)
        return torch.cat((x, y), 1)

class OpsMulti():
    def __init__(self, slices, operators):
        super().__init__()
        self.chs = [0]
        self.ops = []
        total_slice = 0
        for slice, op in zip(slices, operators):
            total_slice += slice * slice
            self.chs.append(total_slice)
            self.ops.append(op(slice))
        
    def __call__(self, x):
        tmp = []
        for i in range(len(self.ops)):
            tmp.append(self.ops[i](x[i]))
        return torch.cat(tmp, 1)
        


class FuseMulti(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.invReorg1 = InvertedReorg(slice = slice[0])
        self.invReorg2 = InvertedReorg(slice = slice[1])
        self.chs1 = slice[0] * slice[0]
        self.chs2 = slice[1] * slice[1]

    def forward(self, x):
        ch = x.shape[1]//(self.chs1 + self.chs2) * self.chs1
        x1 = self.invReorg1(x[:, :ch, :, :])
        x2 = self.invReorg2(x[:, ch:, :, :])
        return torch.cat((x1, x2), 1)

class FusePyramid(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio, slice):
        super().__init__()
        self.conv1 = InvertedResidual(in_chs, out_chs, stride[0], exp_ratio=exp_ratio, slice=slice[0])
        self.invreorg1 = InvertedReorg(slice = slice[0])
        self.reorg1 =  Reorg(slice = slice[0] // 2)
        self.chs1 = slice[0] * slice[0]
        self.conv2 = InvertedResidual(in_chs, out_chs, stride[1], exp_ratio=exp_ratio, slice=slice[1])
        self.invreorg2 = InvertedReorg(slice = slice[1])
        self.chs2 = slice[1] * slice[1]
        self.conv3 = InvertedResidual(in_chs, out_chs, stride[2], exp_ratio=exp_ratio, slice=slice[2])
        self.chs3 = slice[2] * slice[2]

    def forward(self, x):
        ch = x.shape[1] // (self.chs1 + self.chs2 + self.chs3)
        x1 = self.conv1(x[:, :self.chs1*ch, :, :])
        x1 = self.invreorg1(x1)
        x1 = self.reorg1(x1)
        x2 = self.conv2(x[:, self.chs1*ch: (self.chs1 + self.chs2)*ch, :, :])
        x2 = self.invreorg2(x2)
        x3 = self.conv3(x[:, (self.chs1 + self.chs2)*ch:, :, :])
        return torch.cat((x1, x2, x3), 1)


class FusePyramid2(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio, slice):
        super().__init__()
        self.conv1 = InvertedResidual(in_chs, out_chs, stride[0], exp_ratio=exp_ratio, slice=slice[0])
        self.invreorg1 = InvertedReorg(slice = slice[0])
        self.chs1 = slice[0] * slice[0]
        self.conv2 = InvertedResidual(in_chs, out_chs, stride[1], exp_ratio=exp_ratio, slice=slice[1])
        self.chs2 = slice[1] * slice[1]

    def forward(self, x):
        ch = x.shape[1] // (self.chs1 + self.chs2)
        x1 = self.conv1(x[:, :self.chs1*ch, :, :])
        x1 = self.invreorg1(x1)
        x2 = self.conv2(x[:, self.chs1*ch:, :, :])
        return torch.cat((x1, x2), 1)

class ReducePyramid(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio, slice):
        super().__init__()
        self.conv1 = InvertedResidual(in_chs, out_chs, stride[0], exp_ratio=exp_ratio, slice=slice[0])
        self.invreorg1 = InvertedReorg(slice = slice[0])
        self.chs1 = slice[0] * slice[0]
        self.conv2 = InvertedResidual(in_chs, out_chs//4, stride[1], exp_ratio=exp_ratio//2, slice=slice[1])
        self.chs2 = slice[1] * slice[1]
        self.conv3 = InvertedResidual(in_chs, out_chs//16, stride[2], exp_ratio=exp_ratio//2, slice=slice[2])
        self.chs3 = slice[2] * slice[2]

    def forward(self, x):
        ch = x.shape[1] // (self.chs1 + self.chs2 + self.chs3)
        x1 = self.conv1(x[:, :self.chs1*ch, :, :])
        x1 = self.invreorg1(x1)
        x2 = self.conv2(x[:, self.chs1*ch: (self.chs1 + self.chs2)*ch, :, :])
        x3 = self.conv3(x[:, (self.chs1 + self.chs2)*ch:, :, :])
        return torch.cat((x1, x2, x3), 1)

class ReducePyramid3(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio, slice):
        super().__init__()
        self.conv1 = InvertedResidual(in_chs, out_chs, stride[0], exp_ratio=exp_ratio, slice=slice[0])
        self.invreorg1 = InvertedReorg(slice = slice[0])
        self.chs1 = slice[0] * slice[0]
        self.conv2 = InvertedResidual(in_chs, out_chs, stride[1], exp_ratio=exp_ratio//2, slice=slice[1])
        self.chs2 = slice[1] * slice[1]
        self.conv3 = InvertedResidual(in_chs, out_chs, stride[2], exp_ratio=exp_ratio//2, slice=slice[2])
        self.chs3 = slice[2] * slice[2]

    def forward(self, x):
        ch = x.shape[1] // (self.chs1 + self.chs2 + self.chs3)
        x1 = self.conv1(x[:, :self.chs1*ch, :, :])
        x1 = self.invreorg1(x1)
        x2 = self.conv2(x[:, self.chs1*ch: (self.chs1 + self.chs2)*ch, :, :])
        x3 = self.conv3(x[:, (self.chs1 + self.chs2)*ch:, :, :])
        return torch.cat((x1, x2, x3), 1)

class FuseDiffV3(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio, slice):
        super().__init__()
        self.conv1 = InvertedResidual(in_chs, out_chs, stride[0], exp_ratio=exp_ratio, slice=slice[0])
        self.conv2 = InvertedResidual(in_chs, out_chs, stride[1], exp_ratio=exp_ratio, slice=slice[1])
        self.in_chs = out_chs * slice[0] * slice[0]
        self.ops = OpsMulti(slice, [InvertedReorg, AddSlice])

    def forward(self, x):
        x1 = self.conv1(x[:, :self.in_chs, :, :])
        x2 = self.conv2(x[:, self.in_chs:, :, :])
        return self.ops([x1, x2])



class FuseMultiReorg(nn.Module):
    def __init__(self, slice, reorg):
        super().__init__()
        self.invReorg1 = InvertedReorg(slice = slice[0])
        self.invReorg2 = InvertedReorg(slice = slice[1])
        self.chs1 = slice[0] * slice[0]
        self.chs2 = slice[1] * slice[1]
        self.reorg1 = Reorg(reorg[0])
        self.reorg2 = Reorg(reorg[1])

    def forward(self, x):
        ch = x.shape[1]//(self.chs1 + self.chs2) * self.chs1
        x1 = self.invReorg1(x[:, :ch, :, :])
        x2 = self.invReorg2(x[:, ch:, :, :])
        x1 = self.reorg1(x1)
        x2 = self.reorg2(x2)
        return torch.cat((x1, x2), 1)


class FuseDiff(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratios, slice):
        super().__init__()
        self.conv1 = InvertedResidual(in_chs, out_chs, 1, exp_ratio=exp_ratios[0], slice=slice[0])
        self.conv2 = InvertedResidual(in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice[1])
        self.invReorg = InvertedReorg(slice = slice[1])
        self.in_chs = out_chs * slice[0] * slice[0]

    def forward(self, x):
        x1 = self.conv1(x[:, :self.in_chs, :, :])
        x2 = self.conv2(x[:, self.in_chs:, :, :])
        x2 = self.invReorg(x2)
        return torch.cat((x1, x2), 1)

class FuseDiffV2(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratios, slice):
        super().__init__()
        self.conv1 = InvertedResidual(in_chs, out_chs, 1, exp_ratio=exp_ratios[0], slice=slice[0])
        self.conv2 = InvertedResidual(in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice[0])
        self.conv3 = InvertedResidual(in_chs, out_chs, stride, exp_ratio=exp_ratios[0], slice=slice[1])
        self.invReorg = InvertedReorg(slice = slice[1])
        self.in_chs = out_chs * slice[0] * slice[0]

    def forward(self, x):
        x1 = self.conv1(x[:, :self.in_chs, :, :])
        x2 = self.conv2(x[:, :self.in_chs, :, :])
        x2 = self.invReorg(x2)
        x3 = self.conv2(x[:, self.in_chs:, :, :])
        x3 = self.invReorg(x3)
        return torch.cat((x1, x2, x3), 1)
        

class FuseDiffV4(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio, slice):
        super().__init__()
        self.conv1 = InvertedResidual(in_chs, out_chs, 1, exp_ratio=exp_ratio, slice=slice)
        self.conv2 = InvertedResidual(in_chs, out_chs, stride, exp_ratio=exp_ratio, slice=slice)
        self.invReorg = InvertedReorg(slice = slice)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.invReorg(x2)
        return torch.cat((x1, x2), 1)

class FuseOverlap(nn.Module):
    def __init__(self, slice, kernel=2):
        super().__init__()
        self.slice = slice
        self.kernel = kernel
        self.invReorg = InvertedReorg(slice = kernel)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        real_ch = channels // self.slice // self.slice
        patches = []
        for i in range(0, self.slice - self.kernel + 1):
            for j in range(0, self.slice - self.kernel +1):
                patches.append(self.invReorg(self.get_kernel_patches(x, i, j, real_ch)))
        return torch.cat(patches, 1)

    def get_kernel_patches(self, array, i, j, real_ch):
        patches = []
        for x in range(i, i+self.kernel):
            for y in range(j, j+self.kernel):
                start_ch = y*real_ch + x*self.slice*real_ch
                patches.append(array[:, start_ch: start_ch+real_ch, :, :])
        return torch.cat(patches, 1)
    



class AddSlice(nn.Module):
    def __init__(self, patches=None):
        super().__init__()
        if slice:
            self.patches = patches
        else:
            self.patches = 2

    def forward(self, x):
        ch = x.shape[1]//self.patches
        res = x[:, :ch,:,:]
        for i in range(1, self.patches):
            res += x[:, ch*i: ch*i+ch, :, :]
        return  res


class AddSliceDiff(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.ratio = slice[0] * slice[0] // slice[1] // slice[1]
    def forward(self, x):
        ch = x.shape[1]//(self.ratio + 1)*self.ratio
        return x[:, :ch,:,:] + x[:, ch:, :, :].repeat(1, self.ratio, 1, 1)

class AddSliceDiffV2(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.chs1 = slice[0] * slice[0]
        self.chs2 = slice[1] * slice[1]
        self.chs3 = slice[2] * slice[2]
    def forward(self, x):
        ch = x.shape[1]//(self.chs1 + self.chs2 + self.chs3)
        return x[:, :ch*self.chs1,:,:] +\
            x[:, ch*self.chs1:ch*(self.chs1 + self.chs2), :, :].repeat(1, self.chs1 // self.chs2, 1, 1) +\
            x[:, ch*(self.chs1 + self.chs2):, :, :].repeat(1, self.chs1 // self.chs3, 1, 1)


class PoolMulti(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.invreorg = InvertedReorg(slice=slice)
        self.ch = slice * slice
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        ch = x.shape[1]//(self.ch + 1) * self.ch
        x1 = self.invreorg(x[:, :ch, :, :])
        x2 = x[:, ch:, :, :]
        x1 = self.avg_pool(x1)
        x2 = self.avg_pool(x2)
        return torch.cat((x1, x2), 1)


class PoolMulti2(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.ch = slice * slice
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        ch = x.shape[1]//(self.ch + 1) * self.ch
        x1 = x[:, :ch, :, :]
        x2 = x[:, ch:, :, :]
        x1 = self.avg_pool(x1)
        x2 = self.avg_pool(x2)
        return torch.cat((x1, x2), 1)


if __name__ == "__main__":
    # x = torch.range(0, 256*64-1).reshape(1, 1, 64, 256).repeat(1, 3, 4, 1)
    # net = ReorgOverlap(slice=4, overlap=1)
    # x = net(x)
    # print(x.shape)
    # print(x[0, :18:3, :, :])
    
    # x = torch.range(0, 256*3-1).reshape(1, 3, 16, 16)
    # print(x[0, 0, :, :])
    # net = ReorgOverlap(slice=4, overlap=1)
    # x = net(x)
    # print(x.shape)
    # print(x[0, ::3, :, :])
    # x = torch.range(0, 299).reshape(1, 3, 10, 10)
    # net = ReorgOverlap(slice=2.5, overlap=1)
    # x = net(x)
    # print(x.shape)
    # print(x[0, ::3, :, :])
    # net = InvertedReorg(slice=3)
    # x = net(x)
    # print(x.shape)
    # print(x[0, 0, :, :])


    # x = torch.range(0, 220*220-1).reshape(1, 1, 220, 220)
    # net = ReorgOverlap2(patch=64, overlap=12)
    # x = net(x)
    # print(x.shape)
    # print(x[0, -1, :, :])


    x = torch.range(0, 192-1).reshape(1, 3, 8, 8)
    print(x[0, 0, :, :])
    net = ReorgPyramid(slice=[4,2,1])
    x = net(x)
    print(x.shape)
    print(x[0, ::3, :, :])


