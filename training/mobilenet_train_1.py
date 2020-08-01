import argparse
import os
from backbone.MobileNet import mobile_darwinnetv2, mobilenetv2_120, mobile_darwinnet, mobile_darwinnetv3
from util import gpu_data_loader, data_loader, load_pretrained
import torch.nn.utils.prune as prune
from training_adv import fit
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--data", default='/disk3/jinxb/tmp/', metavar='DIR') 
parser.add_argument("--arch","-a",  metavar='ARCH', default='darknet19', help='model architecture: (default: resnet18)')
parser.add_argument("--workers", default=2, type=int, metavar='N')
parser.add_argument("--batch_size", "-b", default=256, type=int)
parser.add_argument("--learning_rate", "-l", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--show_freq", default=10, type=int)
parser.add_argument("--tensorboard", default='runs')
parser.add_argument("--saveboard", action='store_true', default=True)
parser.add_argument("--checkpoint_path", default='checkpoint')
parser.add_argument("--save_all_checkpoint", action='store_true', default=False)
parser.add_argument("--comment", "-m", default="", type=str)
parser.add_argument("--resume","-r", default="", type=str) #checkpoint/2020-06-11T19:45qianyi_mobilenet_relu_lr0.001/epoch_1.pth.tar checkpoint/2020-06-01T22:59darwin_mydata_qianyi/epoch_33.pth.tar
parser.add_argument("--pretrain","-p", default="checkpoint/2020-06-11T22:10qianyi_mobilenet_relu_resume/epoch_4.pth.tar", type=str) #checkpoint/mobilenet/mobilenetv2_120d.pth checkpoint/2020-06-01T16:46darwin_cifar_reduce/epoch_48.pth.tar
parser.add_argument("--evaluate","-e",  dest='evaluate', default=False, action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=12, type=int, help='seed for initializing training. ')
parser.add_argument('--prune', action='store_true', default=False)#checkpoint/2020-06-11T22:10qianyi_mobilenet_relu_resume/epoch_4.pth.tar#checkpoint/2020-06-12T23:28qianyi_mobilenet_darwin/epoch_15.pth.tar
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2"


# #mobil_darwinnetv2
model = mobile_darwinnetv2()
# print(model)
#parallel
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

model = model.cuda()
#train and val data loading
train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=64, workers=args.workers, device_id=0, normalise=True)

# args.pretrain = "checkpoint/epoch_34.pth.tar"
args.resume = "checkpoint/2020-06-18T13:50mobilenet_64_finetune/epoch_6.pth.tar"
lr = 0.01
learning_rate = [lr] * 10 + [lr*0.1] * 10 + [lr*0.01] * 5
remove_layer = ["blocks.1.0"]
need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1.1", "blocks.1.2", "blocks.2", "blocks.3","blocks.4", "blocks.5","conv_head", "bn2", "classifier"]
forzen_epoch = [len(need_forzen)]*6 + [3] *100
fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)