import argparse
import os
from backbone.MoSliceNet import *
from util import load_single_to_multi, read_labels, writeToTxt
from gpu_loader import gpu_data_loader
import torch.nn.utils.prune as prune
from training_adv import fit
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--data", default='/home/jinxb/Project/tmp/', metavar='DIR') 
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
parser.add_argument("--pretrain","-p", default="", type=str) #checkpoint/mobilenet/mobilenetv2_120d.pth checkpoint/2020-06-01T16:46darwin_cifar_reduce/epoch_48.pth.tar
parser.add_argument("--evaluate","-e",  dest='evaluate', default=False, action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=12, type=int, help='seed for initializing training. ')
parser.add_argument('--prune', action='store_true', default=False)#checkpoint/2020-06-11T22:10qianyi_mobilenet_relu_resume/epoch_4.pth.tar#checkpoint/2020-06-12T23:28qianyi_mobilenet_darwin/epoch_15.pth.tar
parser.add_argument('--load_device', default=None)
parser.add_argument('--class_acc', default=None)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2"


# # #mobil_darwinnetv2
# model = moslicenet()

# #parallel
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# # moslicenet()
# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# load_single_to_multi(model, pretrained)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.1] * 10 + [lr*0.01] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4.1", "blocks.4.2", "blocks.4.3", "blocks.4.4", "blocks.5","blocks.6", "conv_head", "bn2", "classifier"]
# forzen_epoch = [len(need_forzen)]*15 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # moslicenetv2
# model = moslicenetv2()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# load_single_to_multi(model, pretrained)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.1] * 10 + [lr*0.01] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


# # moslicenetv3
# model = moslicenetv3()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0":"5.0",
# "6.2":"5.1","6.3":"5.2", "7.0":"5.0","7.2":"5.1","7.3":"5.2", "8.0":"6.0"}
# load_single_to_multi(model, pretrained, repeats=[4*4, 8, 2], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# #mobilenetv3_resume
# model = moslicenetv3()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0, 1]) #多GPU并行
#     # args.load_device = torch.device("cuda:1")

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# args.resume = "checkpoint/2020-06-20T21:48moslicenetv3/epoch_24.pth.tar"
# lr = 0.1
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 10 + [lr*0.002] * 10 
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "blocks.7", "blocks.8", "bn2", "conv_head", "classifier"]
# forzen_epoch = [len(need_forzen)]*20 + [len(need_forzen)-4] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)




# # # moslicenetv5
# model = moslicenetv5()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0":"5.0",
# "6.2":"5.1","6.3":"5.2", "7.0":"5.0","7.2":"5.1","7.3":"5.2", "8.0":"6.0"}
# load_single_to_multi(model, pretrained, repeats=[4*4, 3*3, 2*2], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


# # # moslicenetv5_resume
# model = moslicenetv5()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# args.resume = "checkpoint/2020-06-21T22:40moslicenetv5/epoch_67_24.pth.tar"
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 10
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "blocks.7", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)-3]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # # moslicenetv4
# model = moslicenetv4()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0, 1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0.conv1":"5.0", "6.0.conv2":"5.0", "6.0.conv3":"5.0",
# "6.1":"5.1","6.2":"5.2", "7.0":"5.0","7.2":"5.1","7.3":"5.2", "8.0":"6.0"}
# load_single_to_multi(model, pretrained, repeats=[4*4, 8, 6, 2*2], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


# # # moslicenetv4
# model = moslicenetv6()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0, 1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0":"5.0", 
# "6.1":"5.1","6.2":"5.2", "7.0":"5.0","7.2":"5.1","7.3":"5.2", "8.0":"6.0"}
# load_single_to_multi(model, pretrained, repeats=[32, 8, 4], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)




# moslicenetv3_2
# model = moslicenetv3_2()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# # pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# # rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0.conv1":"5.0","6.0.conv2":"5.0","6.1":"5.1","6.2":"5.2", #"6.0":"5.0","6.2":"5.1","6.3":"5.2",#
# # "7.0":"5.0","7.2":"5.1","7.3":"5.2", "8.0":"6.0"} #"7.1":"5.1","7.2":"5.2"
# # load_single_to_multi(model, pretrained, repeats=[16, 8, 2], rename_layers=rename_layers)
# args.resume = "checkpoint/2020-06-23T20:26moslicenetv3_add_to_cat/epoch_59_1.pth.tar"
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # moslicenetv3_3
# model = moslicenetv3_3()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0":"5.0","6.2":"5.1","6.3":"5.2", #"6.0":"5.0","6.2":"5.1","6.3":"5.2",#
# "7.0":"5.0","7.1":"5.1","7.2":"5.2", "8.0":"6.0"} #"7.1":"5.1","7.2":"5.2"
# load_single_to_multi(model, pretrained, repeats=[16, 4], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # moslicenetv3_3_eval
# model = moslicenetv3_3()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# args.pretrain = "checkpoint/2020-06-23T21:17moslicenetv3_3/epoch_67_22.pth.tar"
# lr = 0.01
# learning_rate = []
# remove_layer = []
# need_forzen = []
# forzen_epoch = []
# args.class_acc = [0] * 1000
# # fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)
# args.class_acc = [43, 41, 39, 28, 40, 37, 35, 38, 42, 48, 40, 44, 45, 44, 41, 44, 40, 41, 39, 45, 38, 34, 42, 43, 45, 46, 27, 35, 37, 42, 37, 36, 18, 32, 30, 23, 25, 38, 35, 37, 23, 39, 36, 35, 30, 32, 25, 38, 40, 29, 32, 43, 34, 40, 30, 29, 40, 41, 26, 29, 16, 34, 20, 35, 27, 36, 28, 31, 14, 42, 40, 45, 47, 19, 27, 41, 44, 34, 36, 31, 39, 35, 42, 46, 43, 43, 35, 46, 44, 44, 47, 41, 44, 46, 46, 45, 38, 36, 41, 34, 46, 26, 50, 39, 34, 42, 35, 39, 34, 44, 36, 46, 36, 37, 34, 40, 36, 39, 39, 26, 31, 37, 40, 36, 21, 34, 27, 42, 37, 48, 46, 43, 43, 42, 31, 48, 46, 39, 45, 47, 40, 40, 40, 44, 43, 45, 40, 42, 43, 41, 39, 25, 35, 38, 36, 29, 41, 42, 25, 34, 40, 39, 43, 15, 41, 23, 29, 15, 32, 41, 22, 36, 35, 30, 43, 26, 33, 41, 44, 34, 23, 43, 37, 32, 30, 34, 33, 32, 34, 32, 37, 37, 34, 24, 40, 40, 30, 30, 36, 35, 28, 30, 31, 42, 30, 39, 34, 41, 35, 41, 43, 34, 39, 37, 41, 36, 42, 44, 43, 36, 34, 40, 25, 39, 35, 43, 24, 33, 42, 42, 35, 25, 40, 35, 36, 37, 34, 35, 28, 43, 15, 26, 33, 34, 42, 39, 25, 42, 23, 37, 22, 48, 34, 43, 47, 50, 36, 36, 47, 47, 45, 44, 43, 40, 34, 23, 29, 31, 39, 38, 40, 24, 29, 34, 42, 50, 39, 28, 30, 45, 34, 30, 8, 41, 41, 20, 45, 31, 41, 43, 38, 40, 46, 46, 47, 36, 41, 36, 30, 39, 45, 44, 27, 31, 32, 40, 38, 42, 35, 35, 32, 30, 26, 30, 33, 27, 42, 42, 36, 32, 41, 47, 44, 46, 43, 43, 45, 41, 40, 32, 37, 40, 48, 47, 43, 29, 39, 36, 37, 47, 49, 22, 41, 43, 41, 25, 36, 44, 32, 34, 43, 43, 41, 27, 43, 39, 20, 32, 27, 32, 37, 45, 39, 42, 45, 39, 37, 39, 37, 33, 39, 30, 41, 36, 24, 40, 49, 34, 37, 42, 25, 18, 32, 37, 42, 28, 31, 46, 47, 36, 30, 38, 42, 39, 34, 30, 47, 40, 35, 35, 20, 40, 23, 38, 44, 34, 32, 45, 30, 34, 42, 32, 23, 22, 18, 20, 39, 42, 23, 25, 42, 34, 29, 31, 27, 47, 40, 26, 30, 36, 44, 39, 38, 30, 26, 19, 25, 35, 19, 44, 32, 32, 29, 25, 37, 21, 27, 30, 41, 35, 44, 34, 29, 32, 28, 29, 34, 27, 46, 26, 29, 17, 27, 24, 24, 25, 47, 38, 29, 20, 25, 40, 48, 27, 32, 43, 46, 32, 20, 20, 33, 29, 8, 38, 33, 21, 36, 33, 12, 32, 33, 42, 30, 13, 24, 38, 42, 30, 32, 16, 47, 11, 29, 30, 22, 28, 36, 35, 31, 27, 42, 36, 38, 22, 31, 31, 23, 36, 46, 29, 39, 37, 43, 21, 22, 34, 20, 26, 41, 30, 28, 29, 38, 46, 22, 48, 22, 41, 35, 30, 41, 18, 20, 24, 30, 39, 30, 42, 46, 20, 32, 36, 40, 42, 46, 41, 26, 39, 15, 32, 42, 41, 38, 36, 38, 47, 37, 15, 34, 46, 39, 37, 32, 44, 40, 38, 46, 37, 31, 34, 39, 36, 30, 38, 19, 21, 36, 22, 33, 31, 23, 40, 36, 34, 43, 36, 18, 34, 34, 32, 11, 26, 34, 39, 39, 36, 36, 46, 32, 32, 39, 40, 45, 41, 40, 37, 32, 36, 14, 28, 10, 37, 23, 16, 30, 45, 27, 33, 44, 36, 40, 31, 24, 16, 35, 27, 30, 38, 14, 16, 46, 29, 43, 20, 33, 48, 36, 31, 28, 40, 23, 20, 28, 30, 37, 24, 19, 21, 31, 35, 25, 45, 25, 24, 16, 31, 37, 25, 49, 49, 37, 39, 43, 18, 32, 24, 23, 26, 25, 42, 36, 12, 38, 30, 40, 46, 29, 43, 44, 9, 35, 21, 24, 34, 46, 34, 32, 25, 30, 38, 24, 46, 31, 35, 41, 28, 32, 37, 29, 31, 22, 34, 30, 43, 31, 38, 45, 27, 24, 37, 34, 36, 38, 38, 38, 22, 40, 44, 19, 26, 35, 19, 40, 18, 38, 24, 39, 28, 40, 44, 19, 34, 23, 29, 18, 34, 42, 32, 17, 33, 29, 29, 34, 33, 29, 47, 36, 38, 29, 34, 32, 33, 20, 38, 15, 28, 40, 22, 39, 37, 39, 39, 17, 27, 25, 24, 31, 36, 30, 44, 37, 47, 4, 44, 19, 38, 37, 24, 37, 40, 27, 38, 26, 30, 32, 34, 31, 30, 34, 20, 44, 40, 42, 43, 18, 44, 27, 32, 30, 22, 18, 25, 43, 14, 45, 35, 42, 28, 13, 33, 45, 44, 39, 27, 25, 37, 22, 28, 34, 43, 26, 33, 42, 34, 26, 28, 11, 24, 18, 33, 36, 13, 24, 34, 30, 28, 26, 35, 20, 32, 37, 22, 37, 41, 42, 27, 28, 31, 35, 30, 17, 26, 31, 47, 24, 31, 37, 34, 27, 24, 38, 36, 36, 46, 46, 29, 18, 38, 34, 28, 36, 35, 34, 25, 37, 8, 47, 25, 40, 22, 46, 32, 25, 25, 33, 36, 39, 40, 21, 10, 47, 27, 32, 40, 30, 24, 12, 21, 25, 25, 20, 19, 32, 41, 36, 44, 49, 33, 43, 34, 38, 26, 41, 20, 35, 27, 39, 45, 27, 26, 20, 27, 33, 41, 35, 35, 40, 44, 41, 35, 36, 37, 35, 32, 40, 42, 45, 17, 42, 34, 35, 37, 38, 40, 41, 45, 41, 42, 38, 41, 21, 12, 30, 42, 33, 36, 30, 27, 14, 20, 29, 35, 29, 30, 46, 26, 24, 32, 21, 35, 34, 33, 35, 39, 49, 48, 50, 19, 44, 45, 41, 47, 44, 47, 40, 50, 39, 34, 26, 17]
# ind_acc = dict()
# for i in range(len(args.class_acc)):
#     ind_acc[i] = args.class_acc[i]
#     # print(args.class_acc[i]/50.0*100, end=" ")

# ordered_acc = sorted(ind_acc.items(),key=lambda x:x[1],reverse=True)
# res = []
# res1 = []
# label_names = read_labels("imagenet_label.txt", name=3)
# for idx in ordered_acc:
#     res.append(label_names[idx[0]])
# for idx in ordered_acc:
#     res1.append(idx[1])


# writeToTxt(res, "rank_list.txt")
# writeToTxt(res1, "acc_list.txt")






# # moslicenetv3_3_resume
# model = moslicenetv3_3()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# args.resume = "checkpoint/2020-06-23T21:17moslicenetv3_3/epoch_67_22.pth.tar"
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 10
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*20 + [len(need_forzen) - 4] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)




# # moslicenetv7
# model = moslicenetv7()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4"} #"7.1":"5.1","7.2":"5.2"
# load_single_to_multi(model, pretrained, repeats=[16, 4], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "bn2", "conv_head", "classifier"]
# forzen_epoch = [len(need_forzen)-3]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


# # moslicenetv8
# model = moslicenetv8()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=1, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0.conv1":"5.0","6.0.conv2":"5.0", "6.1":"5.1","6.2":"5.2", #"6.0":"5.0","6.2":"5.1","6.3":"5.2",#
# "7.0":"5.0","7.1":"5.1","7.2":"5.2", "8.0":"6.0"} 
# load_single_to_multi(model, pretrained, repeats=[16, 5, 4], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.8", "bn2", "conv_head", "classifier"]
# forzen_epoch = [len(need_forzen)-4]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


# # moslicenetv3_4
# model = moslicenetv3_4()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=220, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0":"5.0","6.2":"5.1","6.3":"5.2", #"6.0":"5.0","6.2":"5.1","6.3":"5.2",#
# "7.0":"5.0","7.1":"5.1","7.2":"5.2", "8.0":"6.0"} #"7.1":"5.1","7.2":"5.2"
# load_single_to_multi(model, pretrained, repeats=[16, 4], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # moslicenetv9
# model = moslicenetv9()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.0.conv1":"5.0", "5.0.conv2":"5.0", "5.0.conv3":"5.0", "6.0":"5.0","6.1":"5.1",
# "6.2.conv1":"5.2", "6.2.conv2":"5.2", "6.2.conv3":"5.2","7.0":"5.0","7.1":"5.1","7.2":"5.2","8.0":"6.0"} #"7.1":"5.1","7.2":"5.2"
# load_single_to_multi(model, pretrained, repeats=[21, 16, 6, 4], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)-5]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


# # moslicenetv9_3
# model = moslicenetv9_3()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=256, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.0.conv1":"5.0", "5.0.conv2":"5.0", "5.0.conv3":"5.0", "6.0.conv1":"5.0","6.0.conv2":"5.0","6.0.conv3":"5.0","6.1":"5.1",
# "6.2":"5.2", "7.1":"5.0","7.2":"5.1","7.3":"5.2","8.0":"6.0"} #"7.1":"5.1","7.2":"5.2"
# load_single_to_multi(model, pretrained, repeats=[21, 16, 6, 4, 3], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.8"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # moslicenetv3_6
# model = moslicenetv3_6()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrianed/epoch_19.pth.tar"
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0":"5.0","6.1":"5.1","6.2":"5.2","6.3":"5.3","6.4":"5.4", #"6.0":"5.0","6.2":"5.1","6.3":"5.2",#
# "7.0":"6.0"} #"7.1":"5.1","7.2":"5.2"
# load_single_to_multi(model, pretrained, repeats=[16, 4], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "classifier", "bn2", "conv_head", "blocks.7"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # moslicenetv10
# model = moslicenetv10()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

# # pretrained = "checkpoint/0_pretrained_128/epoch_14.pth.tar"
# # rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4"}
# # load_single_to_multi(model, pretrained, repeats=[4], rename_layers=rename_layers)
# args.resume = "checkpoint/2020-06-28T02:10moslicenetv10_lr0.001/epoch_63_17.pth.tar"
# lr = 0.001
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 15
# remove_layer = []
# need_forzen = []#["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.6", "classifier", "bn2", "conv_head"]
# forzen_epoch = []#[len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


# # moslicenetv10_2
# model = moslicenetv10_2()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

# pretrained = ""
# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0":"5.0","6.1":"5.1","6.2":"5.2"}
# load_single_to_multi(model, pretrained, repeats=[4], rename_layers=rename_layers)
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 5
# remove_layer = []
# need_forzen = ["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.7" "classifier", "bn2", "conv_head"]
# forzen_epoch = [len(need_forzen)]*150 + [9] *100
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


# moslicenetv10_4
# model = moslicenetv10_4()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

# # args.pretrain = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
# args.resume = "checkpoint/2020-07-04T14:29moslicenetv10_4_add_relu/epoch_57_12.pth.tar"
# lr = 0.001
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 15
# remove_layer = []
# need_forzen = []#["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.6", "classifier", "bn2", "conv_head"]
# forzen_epoch = []#[len(need_forzen)]*150 + [9] *100
# # args.evaluate=True
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # moslicenetv10_5
# model = moslicenetv10_5()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

# args.pretrain = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
# lr = 0.01
# learning_rate = [lr] * 10 + [lr*0.1] * 10 + [lr*0.01] * 15
# remove_layer = []
# need_forzen = []#["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.6", "classifier", "bn2", "conv_head"]
# forzen_epoch = []#[len(need_forzen)]*150 + [9] *100
# # args.evaluate=True
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



# # moslicenetv10_nobn
# from backbone.MoSliceNet_NoBN import moslicenetv10_nobn
# from convert import mix_bn_to_conv
# model = moslicenetv10_nobn()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

# # pretrained = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
# pretrained = "checkpoint/2020-07-11T19:27moslicenet_relu_128/epoch_65_33.pth.tar"
# # print(model.state_dict()['module.classifier.weight'])
# name_map = [("blocks.0.0.conv_dw","blocks.0.0.bn1"), ("blocks.0.0.conv_pw","blocks.0.0.bn2"), ("conv_stem", "bn1"), ("conv_head", "bn2"),
#     ("conv_pwl","bn3"), ("conv_pw","bn1"), ("conv_dw", "bn2")]
# mix_bn_to_conv(model, pretrained, conv_names=["conv"], bn_names=["bn"], fc_names=["classifier"], name_map=name_map)#, device=torch.device("cuda:0")
# # print(model.state_dict()['module.classifier.weight'])
# # args.pretrain = pretrained
# lr = 0.001
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 15
# remove_layer = []
# need_forzen = []#["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.6", "classifier", "bn2", "conv_head"]
# forzen_epoch = []#[len(need_forzen)]*150 + [9] *100
# args.evaluate = True
# # for name, param in model.module.named_parameters():
# #     print(name, param.device)
# torch.save({'state_dict': model.state_dict()}, "checkpoint/0_pretrained_128/moslicenet_relu_128_epoch_65_33.pth.tar")
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)





# # moslicenetv10_no_normalisze
# from backbone.MoSliceNet_NoBN import moslicenetv10_nobn
# from backbone.MoSliceNet_Spike import moslicenetv10_spike
# from convert import mix_bn_to_conv
# from validate import validate
# from util import load_pretrained
# model = moslicenetv10_spike()
# model = model.cuda()
# #train and val data loading
# args.batch_size = 256
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)
# pretrained = "checkpoint/2020-07-04T23:11moslicenetv10_4_add_relu_resume/epoch_59_26.pth.tar"
# name_map = [("blocks.0.0.conv_dw","blocks.0.0.bn1"), ("blocks.0.0.conv_pw","blocks.0.0.bn2"), ("conv_stem", "bn1"), ("conv_head", "bn2"),
#     ("conv_pwl","bn3"), ("conv_pw","bn1"), ("conv_dw", "bn2")]
# mix_bn_to_conv(model, pretrained, conv_names=["conv"], bn_names=["bn"], fc_names=["classifier"], name_map=name_map)
# validate(val_loader, model, 400)




# # moslicenetv10_no_normalisze_nobn_max_act
# from backbone.MoSliceNet_NoBN import moslicenetv10_nobn
# from convert import mix_bn_to_conv, save_max_activations
# # from util import load_pretrained
# model = moslicenetv10_nobn(False)

# # #parallell  
# # if torch.cuda.device_count()>1:
# #     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

# # pretrained = "checkpoint/2020-07-04T23:11moslicenetv10_4_add_relu_resume/epoch_59_26.pth.tar"
# # name_map = [("blocks.0.0.conv_dw","blocks.0.0.bn1"), ("blocks.0.0.conv_pw","blocks.0.0.bn2"), ("conv_stem", "bn1"), ("conv_head", "bn2"),
# #     ("conv_pwl","bn3"), ("conv_pw","bn1"), ("conv_dw", "bn2")]
# # mix_bn_to_conv(model, pretrained, conv_names=["conv"], bn_names=["bn"], fc_names=["classifier"], name_map=name_map)
# # lr = 0.01
# args.pretrain = "checkpoint/0_pretrained_128/molicenet_max_act_no_prune.pth.tar"
# learning_rate = []#[lr] * 10 + [lr*0.1] * 10 + [lr*0.01] * 15
# remove_layer = []
# need_forzen = []#["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.6", "classifier", "bn2", "conv_head"]
# forzen_epoch = []#[len(need_forzen)]*150 + [9] *100
# args.evaluate=True
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)
# # save_max_activations(model, "max_act.pth")



# # moslicenetv10_spike_max_act
# from backbone.MoSliceNet_Spike import moslicenetv10_spike
# from validate import validate
# from util import load_pretrained
# model = moslicenetv10_spike()
# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)
# pretrain = "checkpoint/2020-07-04T23:11moslicenetv10_4_add_relu_resume/epoch_59_26.pth.tar"
# load_pretrained(model, pretrain)
# validate(val_loader, model, 400)
# # args.evaluate = True
# # fit(args, model, train_loader, val_loader, [])




# ## moslicenetv10_4_relu_128
# model = moslicenetv10_4()
# print(model)
# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

# pretrained = "checkpoint/0_pretrained_128/mobilenet_relu_128_epoch_20.pth.tar"

# rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4"}
# load_single_to_multi(model, pretrained, repeats=[4], rename_layers=rename_layers)
# lr = 0.001
# learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 15
# remove_layer = []
# need_forzen = []#["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.6", "classifier", "bn2", "conv_head"]
# forzen_epoch = []#[len(need_forzen)]*150 + [9] *100
# # args.evaluate=True
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)


## moslicenetv10_4_relu_128
from backbone.MoSliceNet_Spike import moslicenetv10_spike
from backbone.MoSliceNet_NoBN import moslicenetv10_nobn
# model = moslicenetv10_spike(False) #moslicenetv10_4() #
model = moslicenetv10_nobn(True)
print(model)
# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

model = model.cuda()
#train and val data loading
train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)

args.pretrain = "checkpoint/0_pretrained_128/moslicenet_relu_128_epoch_65_33_nobn.pth.tar"

lr = 0.001
learning_rate = [lr] * 10 + [lr*0.2] * 10 + [lr*0.02] * 15
remove_layer = []
need_forzen = []#["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.6", "classifier", "bn2", "conv_head"]
forzen_epoch = []#[len(need_forzen)]*150 + [9] *100
args.evaluate=True
fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen)



