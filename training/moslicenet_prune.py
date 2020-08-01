import argparse
import os
from backbone.MoSliceNet import *
from util import load_single_to_multi, read_labels, writeToTxt
from gpu_loader import gpu_data_loader
import torch.nn.utils.prune as prune
from training_adv import fit
import torch
import torch.nn as nn
from prune_util import get_parameter_to_prune, PruneSchedualer


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

# # moslicenetv10
# model = moslicenetv10()

# #parallell  
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)


#prune_iteatative
# args.pretrain = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
# # args.resume = "checkpoint/2020-07-01T14:44moslicentv10_prune_itearative/epoch_65_2.pth.tar"
# lr = 0.001

# remove_layer = []
# need_forzen = []#["conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "conv_head", "bn2", "classifier"]
# forzen_epoch = []#[11, 11, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 1, 1] + [0] * 20

# prune_amount = [0.6, 0.8] * 3 + [0.6, 0.8] * 5 + [0.6, 0.8] * 6 + [0.6, 0.7] * 5 + \
#                 [0.6, 0.4] + [0.25, 0.25]*4 + [0.25, 0.2, 0.2, 0.7]
# #6+10+12+10+2+8+4=52
# prune_epoch = [52, 51, 50, 48, 38, 28, 16, 6, 0]
# loops = [2, 6, 6, 6, 4, 4, 4, 4]
# learning_rate = [lr, lr*0.1] + [lr, lr*0.1, lr*0.1, lr*0.1, lr*0.1, lr*0.1 ] * 18 + [lr*0.1] * 16 + [lr*0.1] * 10 + [lr*0.01] * 5
# need_forzen = [["classifier"], ["conv_head", "module.bn2"], ["blocks.6"], ["blocks.5"], ["blocks.4"], ["blocks.3"],["blocks.2"], ["blocks.1"]]
# t1 = [prune_epoch[0]]
# t2 = []
# for i in range(len(loops)):
#     for j in range(loops[i]):
#         t1.append(prune_epoch[i+1])
#         t2.append(need_forzen[i])
# prune_epoch = t1
# need_forzen = t2

# params = get_parameter_to_prune(model, ["conv", "classifier"], ["dw", "stem", "blocks.0"])
# prune_params = []
# for param, amount in zip(params, prune_amount):
#     prune_params.append((param[0], param[1], 1.0 - amount))

# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=remove_layer, forzen_epoch=forzen_epoch, need_forzen=need_forzen, prune_params=prune_params, prune_epoch=prune_epoch)


# #scheduler
# args.pretrain = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
# # args.resume = "checkpoint/2020-07-01T14:44moslicentv10_prune_itearative/epoch_65_2.pth.tar"
# lr = 0.001
# prune_amount = [0.6, 0.8] * 3 + [0.6, 0.8] * 5 + [0.6, 0.8] * 6 + [0.6, 0.7] * 5 + \
#             [0.6, 0.4] + [0.25, 0.25]*4 + [0.25, 0.2, 0.2, 0.7]
# #6+10+12+10+2+8+4=52
# prune_epoch = [52, 51, 50, 48, 38, 28, 16, 6, 0]
# iters = [1, 5, 5, 5, 1, 1, 1, 1]
# learning_rate = [0] * 20 * (1+1) + [lr]*5 + [lr * 0.1] *10 + [lr *0.01]*5
# need_forzen = [["classifier"], ["conv_head", "module.bn2"], ["blocks.6"], ["blocks.5"], ["blocks.4"], ["blocks.3"],["blocks.2"], ["blocks.1"]]
# params = get_parameter_to_prune(model, ["conv", "classifier"], ["dw", "stem", "blocks.0"])
# schedualer = PruneSchedualer(params, prune_amount, prune_epoch, iters, need_forzen, learning_rate=lr, retrain_iter=1)
# fit(args, model, train_loader, val_loader, learning_rate, schedualer=schedualer)


# #prune_iteatative_need_save_checkpoints
# args.pretrain = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
# lr = 0.001
# prune_amount = [0.6, 0.8] * 3 + [0.6, 0.8] * 5 + [0.6, 0.8] * 6 + [0.6, 0.7] * 5 + \
#             [0.6, 0.4] + [0.25, 0.25]*4 + [0.25, 0.2, 0.2, 0.7]
# #6+10+12+10+2+8+4=52
# prune_epoch = [52, 51, 50, 48, 38, 28, 16, 6, 0]
# iters = [1, 5, 5, 5, 1, 1, 1, 1]
# retrain = 1
# learning_rate = [0] * 28 + [lr]*5 + [lr * 0.1] *10 + [lr *0.01]*5
# need_forzen = [["classifier"], ["conv_head", "module.bn2"], ["blocks.6"], ["blocks.5"], ["blocks.4"], ["blocks.3"],["blocks.2"], ["blocks.1"]]
# params = get_parameter_to_prune(model, ["conv", "classifier"], ["dw", "stem", "blocks.0"])
# schedualer = PruneSchedualer(params, prune_amount, prune_epoch, iters, need_forzen, learning_rate=lr, retrain_iter=retrain, retrain_lr=lr)
# fit(args, model, train_loader, val_loader, learning_rate, schedualer=schedualer)



# #prune_norecover
# # args.pretrain = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
# args.resume = "checkpoint/2020-07-02T19:35moslicenet_prune_no_recover/epoch_59_7.pth.tar"
# lr = 0.001
# prune_amount = [0.6, 0.8] * 3 + [0.6, 0.8] * 5 + [0.6, 0.8] * 6 + [0.6, 0.7] * 5 + \
#             [0.6, 0.4] + [0.25, 0.25]*4 + [0.25, 0.2, 0.2, 0.7]
# #6+10+12+10+2+8+4=52
# prune_epoch = [52, 51, 50, 48, 38, 28, 16, 6, 0]
# iters = [1, 1, 1, 1, 1, 1, 1, 1]
# retrain = 0
# learning_rate = [lr] * 10 + [lr * 0.1] *10 + [lr *0.01]*5
# need_forzen = ["conv_stem", "module.bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "conv_head", "module.bn2", "classifier"]
# forzen_epoch = [11, 9, 8, 7, 6, 5, 4, 3] +[0] *50
# params = get_parameter_to_prune(model, ["conv", "classifier"], ["dw", "stem", "blocks.0"])
# schedualer = PruneSchedualer(params, prune_amount, prune_epoch, iters, need_forzen=None, learning_rate=lr, retrain_iter=retrain, retrain_lr=lr)
# fit(args, model, train_loader, val_loader, learning_rate, need_forzen=need_forzen, forzen_epoch=forzen_epoch, schedualer=schedualer)


# #prune_norecover2
# args.pretrain = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
# # args.resume = "checkpoint/2020-07-02T19:35moslicenet_prune_no_recover/epoch_59_7.pth.tar"
# lr = 0.001
# prune_amount = [0.6, 0.8] * 3 + [0.6, 0.8] * 5 + [0.6, 0.8] * 6 + [0.6, 0.7] * 5 + \
#             [0.6, 0.4] + [0.25, 0.25]*4 + [0.25, 0.2, 0.2, 0.7]
# #6+10+12+10+2+8+4=52
# prune_epoch = [52, 51, 50, 48, 38, 28, 16, 6, 0]
# iters = [1, 1, 1, 1, 1, 1, 1, 1]
# retrain = 1
# learning_rate = [lr] * 15 + [lr * 0.1] *10 + [lr *0.01]*5
# need_forzen = ["conv_stem", "module.bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "conv_head", "module.bn2", "classifier"]
# forzen_epoch = [11, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3] +[0] *50
# params = get_parameter_to_prune(model, ["conv", "classifier"], ["dw", "stem", "blocks.0"])
# schedualer = PruneSchedualer(params, prune_amount, prune_epoch, iters, need_forzen=None, learning_rate=lr, retrain_iter=retrain)
# fit(args, model, train_loader, val_loader, learning_rate, need_forzen=need_forzen, forzen_epoch=forzen_epoch, schedualer=schedualer)


#prune_norecover2



# moslicenetv10
model = moslicenetv10_4()
print(model)
#parallell  
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

model = model.cuda()
#train and val data loading
train_loader, val_loader = gpu_data_loader(args.data, batch_size=args.batch_size, img_size=128, workers=args.workers, device_id=2, normalise=True)
args.pretrain = "checkpoint/2020-07-11T19:27moslicenet_relu_128/epoch_65_33.pth.tar"
lr = 0.001
prune_amount = [0.8] + [0.6, 0.8] * 3 + [0.6, 0.8] * 5 + [0.6, 0.8] * 6 + [0.6, 0.7] * 5 + \
            [0.6, 0.4] + [0.25, 0.25]*4 + [0.25, 0.2, 0.2, 0.7]
#1+6+10+12+10+2+8+4=53
prune_epoch = [53, 52, 51, 49, 39, 29, 17, 7, 1, 0]
iters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
retrain = 1
learning_rate = [lr] * 18 + [lr * 0.1] *10 + [lr *0.01]*5
# need_forzen = ["conv_stem", "module.bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "conv_head", "module.bn2", "classifier"]
need_forzen = ["conv_stem", "module.bn1", "blocks.0", "blocks.0.0", "blocks.1", "blocks.1.2", "blocks.2", "blocks.2.4", "blocks.3", 
"blocks.3.5", "blocks.4", "blocks.4.4", "blocks.5", "blocks.5.5", "blocks.6", "blocks.6.0", "conv_head", "module.bn2", "classifier"]
forzen_epoch = [18, 16, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3] +[0] *50
params = get_parameter_to_prune(model, ["conv", "classifier"], ["dw", "blocks.0"])
schedualer = PruneSchedualer(params, prune_amount, prune_epoch, iters, need_forzen=None, learning_rate=lr, retrain_iter=retrain)
fit(args, model, train_loader, val_loader, learning_rate, need_forzen=need_forzen, forzen_epoch=forzen_epoch, schedualer=schedualer)