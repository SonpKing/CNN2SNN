import argparse
import os
from models.VGG import vgg16, vgg16_2
from util import load_single_to_multi, read_labels, writeToTxt, data_loader, get_state, rename_layers
from util.gpu_loader import gpu_data_loader
from util.training_adv import fit
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
from convert import mix_bn_to_conv

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument("--data", default='/home/jinxb/Project/data/cifar-100-python', metavar='DIR') 
parser.add_argument("--data", default='/home/jinxb/Project/tmp/', metavar='DIR')
parser.add_argument("--arch","-a",  metavar='ARCH', default='darknet19', help='model architecture: (default: resnet18)')
parser.add_argument("--workers", default=4, type=int, metavar='N')
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


os.environ["CUDA_VISIBLE_DEVICES"]="2"


# #training on my dataset
# model = vgg16_2()

# # #parallel
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = data_loader(args.data, batch_size=128, img_size=224, workers=args.workers, dataset="imagenet_caffe") 

# args.save_all_checkpoint = True
# name_mapping = {'features.5.weight':'features.4.weight', 'features.5.bias':'features.4.bias',
# 'features.7.weight':'features.6.weight', 'features.7.bias':'features.6.bias', 
# 'features.10.weight':'features.8.weight', 'features.10.bias':'features.8.bias',
# 'features.12.weight':'features.10.weight', 'features.12.bias':'features.10.bias',
# 'features.14.weight':'features.12.weight', 'features.14.bias':'features.12.bias',
# 'features.17.weight':'features.14.weight', 'features.17.bias':'features.14.bias',
# 'features.19.weight':'features.16.weight', 'features.19.bias':'features.16.bias',
# 'features.21.weight':'features.18.weight', 'features.21.bias':'features.18.bias',
# 'features.24.weight':'features.20.weight', 'features.24.bias':'features.20.bias',
# 'features.26.weight':'features.22.weight', 'features.26.bias':'features.22.bias',
# 'features.28.weight':'features.24.weight', 'features.28.bias':'features.24.bias'}
# state = rename_layers("checkpoint/vgg/vgg16.pth", name_mapping, False)
# state = get_state(model, state, [], need_dict=False)
# model.load_state_dict(state)#, strict=False
# # torch.save(model.state_dict(), "checkpoint/vgg/vgg16_2.pth")
# lr = 0.001
# learning_rate = [lr] * 4 + [lr*0.1] * 3 + [lr*0.01] * 3
# # args.evaluate = True
# need_forzen = ['features.0.weight', 'features.0.bias', 'features.2.weight', 'features.2.bias', 
# 'features.6.weight', 'features.6.bias', 'features.10.weight', 'features.10.bias', 
# 'features.12.weight', 'features.12.bias', 'features.16.weight', 'features.16.bias', 
# 'features.18.weight', 'features.18.bias', 'features.22.weight', 'features.22.bias', 
# 'features.24.weight', 'features.24.bias', 'classifier.0.weight', 'classifier.0.bias', 
# 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']
# forzen_epoch = [len(need_forzen)]*100
# fit(args, model, train_loader, val_loader, learning_rate, forzen_epoch=forzen_epoch, need_forzen=need_forzen)




#training on my dataset
model = vgg16_2()

# #parallel
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

model = model.cuda()
#train and val data loading
train_loader, val_loader = data_loader(args.data, batch_size=128, img_size=224, workers=args.workers, dataset="imagenet_caffe") 

args.save_all_checkpoint = True
args.pretrain = "checkpoint/2021-01-06T12:13vgg_caffe_b128_224_image_forzen/epoch_70_10.pth.tar"
lr = 0.0001
learning_rate = [lr] * 2
fit(args, model, train_loader, val_loader, learning_rate)