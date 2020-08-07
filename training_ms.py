import argparse
import os
from models.MobileNet_Slim import *
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

# #prune simulate
# from util.validate import validate
# from util import get_state, load_pretrained
# model = mobilenet_slim_spike(8, False, False, False)
# print(model)
# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# pretrained =  "checkpoint/0/mobilenet_slim_darwin_prune_ok.pth.tar"
# load_pretrained(model, pretrained)
# model = SpikeNet(model, vth=1.0)
# validate(val_loader, model, 200)



# #gen max activation

# model = mobilenet_slim_spike(8, False, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=32, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain =  "checkpoint/0/mobilenet_slim_darwin_prune_ok.pth.tar"
# args.evaluate = True
# fit(args, model, train_loader, val_loader, [])



# #find max activation
# from convert.find_activations import find_activations
# find_activations()


# ##TODO
# # normalisze with max activation
# from convert.find_activations import normalise_max
# from util import load_pretrained
# from convert.convert_mobilenet_slim import normalise_module
# from util.validate import validate
# model = mobilenet_slim_spike(8, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=32, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain =  "checkpoint/0/mobilenet_slim_darwin_prune_ok.pth.tar"
# load_pretrained(model, args.pretrain, [])
# max_act = normalise_max(model, "max_activations_res/2020-08-04T21:45")
# normalise_module(model, "", max_act, 1.0, 1.0)
# torch.save({"state_dict": model.state_dict()}, "checkpoint/0/mobilenet_slim_darwin_normalise_2.pth.tar")
# validate(val_loader, model, 350)
# print(model)



# %% find scale
# import torch
# state_path = "checkpoint/0/mobilenet_slim_darwin_prune_ok_normalise.pth.tar"
# from util import get_state
# state = torch.load(state_path)['state_dict']
# for key in state:
#     state[key] = torch.round(state[key]*100)
#     state[key][state[key]>127] = 127
#     state[key][state[key]<-127] = -127
# data = []
# for key in state:
#     data += state[key].cpu().numpy().ravel().tolist()
# data.sort()
# length = len(data)
# print(data[5], data[int(length*0.0001)], data[int(length*0.1)], data[int(length*0.9)], data[int(length*0.9999)], data[-5])
# torch.save({"state_dict": state}, "checkpoint/0/mobilenet_slim_darwin_prune_ok_normalise_scale_100.pth.tar")


# ## validate
# from util import load_pretrained
# from util.validate import validate
# from models import SpikeNet
# model = mobilenet_slim_spike(8, False, True, vth=50.0)
# model = model.cuda()
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain = "checkpoint/0/mobilenet_slim_darwin_prune_ok_normalise_scale_50.pth.tar"
# load_pretrained(model, args.pretrain, [])
# model = SpikeNet(model, vth=50.0)
# validate(val_loader, model, 300)


# # generate connections
# from util import load_pretrained
# from convert.convert_mobilenet_slim import convert_module
# from convert import mute_prune_connections
# model = mobilenet_slim_spike(8, False, True, vth=50.0)
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain = "checkpoint/0/mobilenet_slim_darwin_prune_ok_normalise_scale_50.pth.tar"
# load_pretrained(model, args.pretrain, [])
# convert_module(model, "net", 1, "input", (3, 32, 32), prune=False)
# # mute_prune_connections("connections", "connections_new")