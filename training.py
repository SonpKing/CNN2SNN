import argparse
import os
from models.AppleNet import *
from util import load_single_to_multi, read_labels, writeToTxt, data_loader
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

# #training with cifar
# model = applenetv1(100)

# #parallel
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = data_loader(args.data, batch_size=args.batch_size, img_size=32, workers=args.workers, dataset="cifar") 
# lr = 0.1
# learning_rate = [lr] * 60 + [lr*0.1] * 40 + [lr*0.01] * 20 + [lr*0.001] * 10
# fit(args, model, train_loader, val_loader, learning_rate)


# # no_norm
# model = applenetv2(100, True)
# #parallel
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行
# model.cuda()
# train_loader, val_loader = data_loader(args.data, batch_size=args.batch_size, img_size=32, workers=args.workers, dataset="cifar") 
# pretrained = "checkpoint/2020-07-30T19:35applenet_cifar_no_last_at/epoch_52_127.pth.tar"
# name_map = [("conv","bn")]
# mix_bn_to_conv(model, pretrained, conv_names=["conv"], bn_names=["bn"], fc_names=["fc"], name_map=name_map)
# torch.save({"state_dict": model.state_dict()}, "checkpoint/0/applenet_cifar_no_norm.pth.tar")
# lr = 0.01
# learning_rate = [lr] * 30 + [lr*0.1] * 20
# args.evaluate=True
# fit(args, model, train_loader, val_loader, learning_rate)



# # no_bias
# model = applenetv2(100, False)
# #parallel
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行
# model.cuda()
# train_loader, val_loader = data_loader(args.data, batch_size=args.batch_size, img_size=32, workers=args.workers, dataset="cifar") 
# args.pretrain = "checkpoint/0/applenet_cifar_no_norm.pth.tar"
# # args.resume = "checkpoint/2020-07-30T19:21applenet_cifar_no_bias/epoch_48_50.pth.tar"
# lr = 0.01
# learning_rate = [lr] * 30 + [lr*0.1] * 20 + [lr*0.01] * 10
# fit(args, model, train_loader, val_loader, learning_rate)




# # trainning on my data
# model = applenetv2(8, False)
# # print(model)
# # exit(0)
# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=32, img_size=32, workers=args.workers, dataset="imagenet") 
# lr = 0.01
# args.pretrain = "checkpoint/2020-07-30T19:52applenet_no_last_act_no_bias/epoch_51_56.pth.tar"
# learning_rate = [lr] * 20 + [lr*0.1] * 10 + [lr*0.01] * 10

# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=["fc"])



# #prune simulate
# from prune_util import get_parameter_to_prune, PruneSchedualer
# from convert import convert_prune_weight
# from validate import validate
# from util import get_state, load_pretrained
# model = applenetv2(8, False, True, False)
# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=32, img_size=32, workers=args.workers, dataset="imagenet") 
# pretrained =  "checkpoint/2020-07-30T19:59applenet_mydata_no_last_act/epoch_80_20.pth.tar"
# load_pretrained(model, pretrained)
# validate(val_loader, model, 300)





# #gen max activation

# model = applenetv2(8, False, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=32, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain =  "checkpoint/2020-07-30T19:59applenet_mydata_no_last_act/epoch_80_20.pth.tar"
# args.evaluate = True
# fit(args, model, train_loader, val_loader, [])




# #find max activation
# from find_activations import find_activations
# find_activations()


# # normalisze with max activation
# from find_activations import normalise_max
# from util import load_pretrained
# from convert import normalise_module
# from validate import validate
# model = applenetv2(8, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# # args.pretrain =  "checkpoint/0/mobilenet_slim_darwin_prune_ok.pth.tar"
# args.pretrain =  "checkpoint/2020-07-30T19:59applenet_mydata_no_last_act/epoch_80_20.pth.tar"
# load_pretrained(model, args.pretrain, [])
# max_act = normalise_max(model, "max_activations_res/2020-07-30T20:19")
# normalise_module(model, "", max_act, 1.0, 1.0)
# torch.save({"state_dict": model.state_dict()}, "checkpoint/0/applenet_normalise.pth.tar")
# validate(val_loader, model, 350)
# print(model)



# %% find scale
# import torch

# state_path = "checkpoint/0/applenet_normalise.pth.tar"
# from util import get_state
# state = torch.load(state_path)['state_dict']
# # import numpy as np  #导入numpy包，用于生成数组
# # import seaborn as sns  #习惯上简写成snssns.set()      
# # import pickle     
# # sns.set()#切换到seaborn的默认运行配置
# # sns.kdeplot(data,color="g")
# for key in state:
#     state[key] = torch.round(state[key]*100)
#     state[key][state[key]>127] = 127
#     state[key][state[key]<-127] = -127

# data = []
# for key in state:
#     data += state[key].cpu().numpy().ravel().tolist()
# data.sort()
# length = len(data)
# print(data[0], data[int(length*0.0001)], data[int(length*0.1)], data[int(length*0.9)], data[int(length*0.9999)], data[-1])
# torch.save({"state_dict": state}, "checkpoint/0/applenet_normalise_scale.pth.tar")


# validate
from util import load_pretrained
from convert.convert_applenet import normalise_module, convert_module
from util.validate import validate
model = applenetv2_spike(8, vth=100.0)
model = model.cuda()
train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
args.pretrain = "checkpoint/0/applenet_normalise_scale.pth.tar"
load_pretrained(model, args.pretrain, [])
validate(val_loader, model, 300)


# # generate connections
# from util import load_pretrained
# from convert import normalise_module, convert_module, mute_prune_connections
# model = applenetv2_spike(8, vth=100.0)
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain = "checkpoint/0/applenet_normalise_scale.pth.tar"
# load_pretrained(model, args.pretrain, [])
# convert_module(model, "net", 1, "input", (3, 32, 32), prune=False)
# mute_prune_connections("connections", "connections_new")


# %%
