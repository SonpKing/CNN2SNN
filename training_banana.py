import argparse
import os
from models.BananaNet import *
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
# model = banananetv1(100)

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
# model = banananetv2(100, True)
# #parallel
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行
# model.cuda()
# train_loader, val_loader = data_loader(args.data, batch_size=args.batch_size, img_size=32, workers=args.workers, dataset="cifar") 
# pretrained = "checkpoint/2020-08-07T11:29banananet_cifar/epoch_52_117.pth.tar"
# name_map = [("conv","bn")]
# mix_bn_to_conv(model, pretrained, conv_names=["conv"], bn_names=["bn"], fc_names=["fc"], name_map=name_map)
# torch.save({"state_dict": model.state_dict()}, "checkpoint/0/banananet_cifar_no_norm.pth.tar")
# args.evaluate=True
# fit(args, model, train_loader, val_loader, learning_rate)



# # no_bias
# model = banananetv2(100, False)
# #parallel
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行
# model.cuda()
# train_loader, val_loader = data_loader(args.data, batch_size=args.batch_size, img_size=32, workers=args.workers, dataset="cifar") 
# args.pretrain = "checkpoint/0/banananet_cifar_no_norm.pth.tar"
# lr = 0.01
# learning_rate = [lr] * 30 + [lr*0.1] * 20 + [lr*0.01] * 10
# fit(args, model, train_loader, val_loader, learning_rate)




# # trainning on my data
# model = banananetv2(7, False)
# # print(model)
# # exit(0)
# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Detect_Data", batch_size=16, img_size=32, workers=args.workers, dataset="imagenet") 
# lr = 0.01
# args.pretrain = "checkpoint/2020-08-07T12:10banananet_cifar_no_bias/epoch_51_59.pth.tar"
# learning_rate = [lr] * 40 + [lr*0.1] * 20 + [lr*0.01] * 20

# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=["fc2"])



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
# #you should add act for your last layer  and every pool layer firstly
# model = banananetv2(7, False, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Detect_Data", batch_size=32, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain =  "checkpoint/2020-08-07T13:29banananet_mydata_2/epoch_84_50.pth.tar"
# args.evaluate = True
# fit(args, model, train_loader, val_loader, [])




# #find max activation
# from convert.find_activations import find_activations
# find_activations()


# # normalisze with max activation
# from convert.find_activations import normalise_max
# from util import load_pretrained
# from convert.convert_banananet import normalise_module
# from util.validate import validate
# model = banananetv2(7, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Detect_Data", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain =  "checkpoint/2020-08-07T13:29banananet_mydata_2/epoch_84_50.pth.tar"
# load_pretrained(model, args.pretrain, [])
# max_act = normalise_max(model, "max_activations_res/2020-08-07T14:28")
# normalise_module(model, "", max_act, 1.0, 1.0)
# torch.save({"state_dict": model.state_dict()}, "checkpoint/0/banananet_normalise.pth.tar")
# validate(val_loader, model, 350)
# print(model)



# %% find scale
# import torch

# state_path = "checkpoint/0/banananet_normalise.pth.tar"
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
# print(data[3], data[int(length*0.0001)], data[int(length*0.1)], data[int(length*0.9)], data[int(length*0.9999)], data[-3])
# torch.save({"state_dict": state}, "checkpoint/0/banananet_normalise_scale100.pth.tar")


# # validate
# from util import load_pretrained
# from convert.convert_applenet import normalise_module, convert_module
# from util.validate import validate
# from models import SpikeNet
# model = banananetv2_spike(7, vth=100.0)
# model = model.cuda()
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Detect_Data", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain = "checkpoint/0/banananet_normalise_scale100.pth.tar"
# load_pretrained(model, args.pretrain, [])
# # model = SpikeNet(model, vth=100.0)
# print(model)
# validate(val_loader, model, 120)


# # generate connections
# from util import load_pretrained
# from convert.convert_banananet import convert_module
# model = banananetv2_spike(7, vth=100.0)
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Detect_Data", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain = "checkpoint/0/banananet_normalise_scale100.pth.tar"
# load_pretrained(model, args.pretrain, [])
# convert_module(model, "net", 1, "input", (3, 32, 32), prune=False)
# # mute_prune_connections("connections", "connections_new")



# # generate connections
# from util import load_pretrained
# # from convert import normalise_module, convert_module, mute_prune_connections
# from convert.cnnconn import conv2d_connections, avg_pool_connections, fc_connections
# from convert import save_connections, shuffle_for_view
# model = banananetv2_spike(7, vth=100.0)
# train_loader, val_loader = data_loader("/home/jinxb/Project/data/Detect_Data", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain = "checkpoint/0/banananet_normalise_scale100.pth.tar"
# load_pretrained(model, args.pretrain, [])
# parameters = []
# view = True
# for name, params in model.named_parameters():
#     if "conv" in name:
#         parameters.append(params.data.numpy().transpose((2, 3, 1, 0)))
#     elif "fc" in name:
#         if view:
#             params.data = shuffle_for_view(params.data, (28, 4, 4))
#             view = False
#         parameters.append(params.data.numpy().transpose((1, 0)))
#     else:
#         parameters.append(params.data.numpy())
# for param in parameters:
#     print(param.shape)
# DELAY = 0
# input_block0conv = conv2d_connections((32, 32, 3), parameters[0], (1, 1),(1, 1), DELAY)
# block0conv_block0pool = avg_pool_connections((32, 32, 14), (2, 2), (0, 0), (2, 2), DELAY, np.round(parameters[1][0]/(2 * 2)))
# block0pool_block1conv = conv2d_connections((16, 16, 14), parameters[2], (1, 1),(1, 1), DELAY)
# block1conv_block1pool = avg_pool_connections((16, 16, 22), (2, 2), (0, 0), (2, 2), DELAY, np.round(parameters[3][0]/(2 * 2)))
# block1pool_block2conv = conv2d_connections((8, 8, 22), parameters[4], (1, 1),(1, 1), DELAY)
# block2conv_block2pool = avg_pool_connections((8, 8, 28), (2, 2), (0, 0), (2, 2), DELAY, np.round(parameters[5][0]/(2 * 2)))
# block2pool_fc1 = fc_connections(parameters[6], DELAY)
# fc1_fc2 = fc_connections(parameters[7], DELAY)
# save_connections(input_block0conv, "input_to_net.blocks.0.conv_chip0")
# save_connections(block0conv_block0pool, "net.blocks.0.conv_to_net.blocks.0.pool_chip0")
# save_connections(block0pool_block1conv, "net.blocks.0.pool_to_net.blocks.1.conv_chip0")
# save_connections(block1conv_block1pool, "net.blocks.1.conv_to_net.blocks.1.pool_chip0")
# save_connections(block1pool_block2conv, "net.blocks.1.pool_to_net.blocks.2.conv_chip0")
# save_connections(block2conv_block2pool, "net.blocks.2.conv_to_net.blocks.2.pool_chip0")
# save_connections(block2pool_fc1, "net.blocks.2.pool_to_net.fc1_chip0")
# save_connections(fc1_fc2, "net.fc1_to_net.fc2_chip0")


# %%
