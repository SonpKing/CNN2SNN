import argparse
import os
from models.MobileNet_Slim import *
from util import load_single_to_multi, read_labels, writeToTxt, data_loader_anno, data_loader_anno
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

# # # # #training on my dataset
# model = mobilenet_slim_2(15, False)

# #parallel
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1]) #多GPU并行

# model = model.cuda()
# #train and val data loading
# train_loader, val_loader = data_loader_anno("/home/jinxb/Project/data/Detect_Annoed", batch_size=16, img_size=32, workers=args.workers, dataset="imagenet") 
# lr = 0.01
# args.save_all_checkpoint = True
# # args.pretrain = "checkpoint/mobile_slim/epoch_51_39.pth.tar"
# #"checkpoint/2020-08-28T18:00slim_anno_nice7_pretrain/epoch_81_18.pth.tar"
# args.resume = "checkpoint/2020-08-28T21:17slim_anno_loss_weight_7/epoch_83_80.pth.tar"
# learning_rate = [lr] * 40 + [lr*0.1] * 20 + [lr*0.01] * 25 + [lr*0.001] * 10#
# loss_weight = [1.0]*14
# loss_weight[11] = 2.0
# fit(args, model, train_loader, val_loader, learning_rate, rm_layers=["classifier"], annoed=True, loss_weight=loss_weight)


# # clear
# paths = ["max_activations", "max_activations_res", "connections"]
# for path in paths:
#     if os.path.exists(path):  
#         for root, dirs, files in os.walk(path, topdown=False):
#             for name in files:
#                 os.remove(os.path.join(root, name))
#             for name in dirs:
#                 os.rmdir(os.path.join(root, name))
#         os.removedirs(path)
#         print("remove", path)
#     else:
#         print('no such file:%s'%path)


#gen max activation

# model = mobilenet_slim_spike(15, False, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader_anno("/home/jinxb/Project/data/Detect_Annoed", batch_size=32, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain =  "checkpoint/2020-08-28T21:27slim_anno_loss_weight_7_resume/epoch_83_87.pth.tar"#"checkpoint/2020-08-28T18:17slime_annp_nice7_pretrain_resume/epoch_81_20.pth.tar"#
# args.evaluate = True
# fit(args, model, train_loader, val_loader, [])

# def read_data(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data.cpu().numpy()

# tmp = []
# for i in range(5):
#     data = read_data("max_activations/20/"+str(i+1))
#     print(data.shape)
#     tmp.extend(data[:, -1])
# print(tmp)


# # #find max activation
# from convert.find_activations import find_activations
# find_activations()


# ##TODO
############## normalisze with max activation
# from convert.find_activations import normalise_max
# from util import load_pretrained
# from convert.convert_mobilenet_slim import normalise_module
# from util.validate import validate
# model = mobilenet_slim_spike(15, False, True)
# model = model.cuda()
# train_loader, val_loader = data_loader_anno("/home/jinxb/Project/data/Detect_Annoed", batch_size=32, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain =  "checkpoint/2020-08-28T21:27slim_anno_loss_weight_7_resume/epoch_83_87.pth.tar"
# load_pretrained(model, args.pretrain, [])
# max_act = normalise_max(model, "max_activations_res")
# normalise_module(model, "", max_act, 1.0, 1.0)
# torch.save({"state_dict": model.state_dict()}, "checkpoint/0/slim_anno_loss_weight_normalise.pth.tar")
# validate(val_loader, model, 350)
# print(model)


# %% find scale
# import torch
# state_path =  "checkpoint/0/slim_anno_loss_weight_normalise.pth.tar"
# from util import get_state
# state = torch.load(state_path)['state_dict']
# for key in state:
#     state[key] = torch.round(state[key]*70)
#     state[key][state[key]>127] = 127
#     state[key][state[key]<-127] = -127
# data = []
# for key in state:
#     data += state[key].cpu().numpy().ravel().tolist()
# data.sort()
# length = len(data)
# print(data[5], data[int(length*0.0001)], data[int(length*0.1)], data[int(length*0.9)], data[int(length*0.9999)], data[-5])
# torch.save({"state_dict": state}, "checkpoint/0/slim_anno_loss_weight_normalise_scale70.pth.tar")


# # ## validate
from util import load_pretrained
from util.validate import validate
from models import SpikeNet
model = mobilenet_slim_spike(15, vth=60.0)
model = model.cuda()
train_loader, val_loader = data_loader_anno("/home/jinxb/Project/data/Detect_Annoed", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
args.pretrain = "checkpoint/0/slim_anno_loss_weight_normalise_scale60.pth.tar"
load_pretrained(model, args.pretrain, [])
model = SpikeNet(model, vth=60)
conf = []
validate(val_loader, model, 500, annoed=True, conf=conf)
print(conf)


# # # # # # # generate connections
# from util import load_pretrained
# from convert.convert_mobilenet_slim import convert_module
# from convert import mute_prune_connections
# model = mobilenet_slim_spike(15, False, True, vth=60.0)
# train_loader, val_loader = data_loader_anno("/home/jinxb/Project/data/Detect_Annoed", batch_size=1, img_size=32, workers=args.workers, dataset="imagenet") 
# args.pretrain = "checkpoint/0/slim_anno_loss_weight_normalise_scale60.pth.tar"
# load_pretrained(model, args.pretrain, [])
# convert_module(model, "net", 1, "input", (3, 32, 32), prune=False)
# # mute_prune_connections("connections", "connections_new")




#%% test dataloader
# train_loader, val_loader = data_loader_anno("/home/jinxb/Project/data/Detect_Annoed", batch_size=16, img_size=32, workers=args.workers, dataset="imagenet") 
# for inputs, targets, conf in train_loader:
#     print(targets, conf)
# for inputs, targets in val_loader:
#     print(targets)