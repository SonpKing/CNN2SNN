import argparse
import os
from models.MobileNet_Slim import *
from util import load_single_to_multi, read_labels, writeToTxt, data_loader_darwin, data_loader_anno
from util.gpu_loader import gpu_data_loader
from util.training_adv import fit
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
from convert import mix_bn_to_conv

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--data", default='/home/jinxb/Project/data/cifar-100-python', metavar='DIR') 
# parser.add_argument("--data", default='/home/jinxb/Project/tmp/', metavar='DIR')
parser.add_argument("--arch","-a",  metavar='ARCH', default='darknet19', help='model architecture: (default: resnet18)')
parser.add_argument("--workers", default=2, type=int, metavar='N')
parser.add_argument("--batch_size", "-b", default=128, type=int)
parser.add_argument("--learning_rate", "-l", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--show_freq", default=10, type=int)
parser.add_argument("--tensorboard", default='runs')
parser.add_argument("--saveboard", action='store_true', default=True)
parser.add_argument("--checkpoint_path", default='checkpoint')
parser.add_argument("--save_all_checkpoint", action='store_true', default=False)
parser.add_argument("--comment", "-m", default="", type=str)
parser.add_argument("--resume","-r", default="", type=str) 
parser.add_argument("--pretrain","-p", default="", type=str) 
parser.add_argument("--evaluate","-e",  dest='evaluate', default=False, action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=12, type=int, help='seed for initializing training. ')
parser.add_argument('--prune', action='store_true', default=False)
parser.add_argument('--load_device', default=None)
parser.add_argument('--class_acc', default=None)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2"


# model = mobilenet_slim_spike(14, False, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader_darwin("/home/jinxb/Project/data/Detect_Data", batch_size=1, img_size=64, workers=1, dataset="imagenet") 
# args.pretrain =  "checkpoint/2020-08-26T18:16slim_nice7/epoch_84_53.pth.tar"
# args.evaluate = True
# fit(args, model, train_loader, val_loader, [])


#！！！！！！！！！！！！！！！
'''
self.pool = nn.AvgPool2d(8)
'''

import matplotlib.pyplot as plt
def read_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data.cpu().numpy()

a = [0, 2, 5,8, 10, 18]
b = [[4,7,17,24],
[1,7,8,23],
[10,18,24,29],
[3,10,11,26],
[1,3,5,6],
[2,3,5,7]]
i = 0
k = 0
for j in a:
    data = read_data("max_activations/"+str(j)+"/11")
    for i in b[k//4]:
        ax = plt.subplot(4, 6, (k%4)*6+k//4 + 1)
        ax.axis('off')
        if(k%4==3):
            ax.set_xlabel(str(j+1))
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            for key, spine in ax.spines.items():
                # 'left', 'right', 'bottom', 'top'
                spine.set_visible(False)
        
        plt.imshow(data[0,i-1,:,:],cmap='jet')
        k = k + 1
plt.savefig("thesis/0.pdf")    
# for j in range(21): 
#     # data = read_data("max_activations/"+str(j)+"/11")
#     data = read_data("max_activations/"+str(j)+"/11")
#     for i in range(min(data.shape[1],36)):  # 可视化了32通道
#         ax = plt.subplot(6, 6, i + 1)
#         ax.axis('off')
#         plt.imshow(data[0,i,:,:],cmap='jet')

#     plt.savefig("thesis/"+str(j)+".png")
#     print(j)



# for i in range(140):  # 可视化了32通道
#     data = read_data("max_activations/2/"+str(i+1))
#     ax = plt.subplot(12, 12, i + 1)
#     ax.axis('off')
#     plt.imshow(data[0,0,:,:],cmap='jet')
# plt.savefig("thesis/"+str(0)+".png")
# from torchsummary import summary
# import torch.backends.cudnn as cudnn
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# cudnn.benchmark = True
# net = mobilenet_slim_spike(14, False, False, False).cuda()
# # for _ in range(10):
# #     input = torch.randn((1, 3, 128, 128)).cuda()
# #     print(net(input))
# summary(net, (3, 64, 64))