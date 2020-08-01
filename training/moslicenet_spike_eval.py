#test
from backbone.MoSliceNet_NoBN import moslicenetv10_nobn
from backbone.MoSliceNet_Spike import moslicenetv10_spike, delay_bias, remove_bias
import torch
from util import load_pretrained
from torch.autograd import Variable
from gpu_loader import gpu_data_loader


def net_forward(net, x):
    x = net.reorg(x)
    x = net.conv_stem(x)
    x = net.act1(x)
    x = net.blocks[0](x)
    x = net.blocks[1](x)
    x = net.blocks[2](x)
    x = net.blocks[3](x)
    x = net.blocks[4](x)
    x = net.blocks[5](x)
    x = net.blocks[6](x)
    x = net.conv_head(x)
    x = net.act2(x)
    x = net.global_pool(x)
    x = net.act3(x)
    x = x.view(x.shape[0], -1)
    x = net.classifier(x)
    x = net.act4(x)
    return x


pretrained1 = "checkpoint/0_pretrained_128/moslicenet_relu_128_epoch_65_33_nobn.pth.tar"
pretrained2 = "checkpoint/0_pretrained_128/moslicenet_relu_128_epoch_65_33_nobn_compensation.pth.tar"
# pretrained = "checkpoint/2020-07-04T23:11moslicenetv10_4_add_relu_resume/epoch_59_26.pth.tar"
net1 = moslicenetv10_spike(False).cuda()
net2 = moslicenetv10_spike().cuda()
load_pretrained(net1, pretrained1)
load_pretrained(net2, pretrained1)
# remove_bias(net1)
# remove_bias(net2)
net1.eval()
net2.eval()
# print(net1.blocks[0][0].conv_pw.weight == net1.blocks[0][0].conv_pw.weight)
# delay_bias(net2)
# print((net1.blocks[0][0].conv_dw.weight==net2.blocks[0][0].conv_dw.weight).all())
# inputs = Variable(torch.randn((1, 3, 128, 128))).cuda()
train_loader, val_loader = gpu_data_loader('/home/jinxb/Project/tmp/', batch_size=8, img_size=128, workers=1, device_id=2, normalise=True)

from backbone.MoSliceNet_Spike import reset_spikenet
with torch.no_grad():
    for it, (inputs, targets) in enumerate(val_loader):
        # load_pretrained(net2, pretrained1)
        reset_spikenet(net2)
        inputs = torch.autograd.Variable(inputs).cuda(non_blocking=True)
        # inputs = Variable(torch.randn((1, 3, 128, 128))).cuda()
        # output1 = net1(inputs)
        output1 = net_forward(net1, inputs)

        output2 = torch.zeros_like(output1)
        
        tics = 5000
        for _ in range(tics):
            output2 += net_forward(net2, inputs)

        # remove_bias(net2)
        # inputs = Variable(torch.zeros((1, 3, 128, 128))).cuda()
        # for _ in range(tics*2):
        #     output2 += net_forward(net2, inputs)

            # output2 += net2(inputs)
            # print(torch.max(output2))
            # net2(inputs)

        # print(output1[0,:4]*100, net2.blocks[-1][-1].act3.total_num[0,:4])
        # output2 = net2.act4.total_num
        print(torch.argmax(output1), torch.argmax(output2), torch.argmax(output1)==torch.argmax(output2))
        print(torch.max(output1)*tics, torch.max(output2))
        print(torch.max(torch.abs(output1*tics-output2)))
        if it >=10: break

# print(output1[0,:4]*100, net2.blocks[0][0].act2.total_num[0,:4])
# print(net2.blocks[0][0].act2.total_mem[0,:4])
#blocks[-1][-1].act3, 