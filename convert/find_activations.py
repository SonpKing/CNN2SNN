import numpy as np
import pickle
import os
from collections import Counter
from multiprocessing import Process
from multiprocessing import Pool



def read_data(file_path):   
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data.cpu().numpy().ravel() 


def save_data(file_path, obj):   
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def find_range(files):
    print("start find max value")
    maxv = float('-inf')
    total_num = 0
    for file in files:
        data = read_data(file)
        maxv = max(maxv, np.max(data))
        total_num += data.size
        # print(file, maxv, total_num)
    return maxv, total_num
        

def find_bins(files, minv, maxv, pivot_num, bin_num=10000):
    one_bin = (maxv-minv)/(bin_num-1)  #bin range from 0 to bin_num
    res = {i:0 for i in range(bin_num)}
    for file in files:
        data = read_data(file)
        data = ((data - minv) / one_bin).astype(np.int)
        count = Counter(data)
        for key in count:
            if 0<= key < bin_num:
                res[key] += count[key]
        # print(res[0], res[bin_num-1])
    sum = 0
    idx = 0
    for i in range(bin_num):
        sum += res[i]
        if sum >= pivot_num:
            sum -= res[i]
            idx = i
            break
    return minv + one_bin * idx, minv + one_bin * (idx + 1), pivot_num - sum
    

def normalise_max(net, path):
    name_param = net.named_parameters()
    idx = 0
    res = dict()
    for layer, _ in name_param:
        if "scale" in layer:
            continue
        layer = ".".join(layer.split('.')[:-1])
        if layer not in res:
            file_path = os.path.join(path, str(idx)+"_max_act")
            idx += 1
            with open(file_path, 'rb') as f:
                act = pickle.load(f)
                print(idx, act)
            res[layer] = (act[0] + act[1]) / 2.0
    print("total max_act", idx)
    return res

import os,shutil

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s"%( srcfile,dstfile))

def move_max_act(path, save_path_dir):
    files = os.listdir(path)
    for i in range(len(files)):
        srcpath = os.path.join(path, files[i], "max_act")
        despath = os.path.join(save_path_dir, files[i]+"_max_act")
        mymovefile(srcpath, despath)
    # move_max_act(path, "max_activations_res/2020-07-20T16:02")
   
    
def find_max_act(params, ratio=1):
    path, save_path = params
    files = os.listdir(path)
    for i in range(len(files)):
        files[i] = os.path.join(path, files[i])
    file_len = len(files) 
    print("process", path, ", total", file_len, "files")
    maxv, total_num = find_range(files)
    minv = 0
    tail = file_len*32
    pivot_num = int(total_num - tail)
    while maxv - minv > 1e-5 and pivot_num > 10:
        print("process", path, minv, maxv, pivot_num)
        minv, maxv, pivot_num = find_bins(files, minv, maxv, pivot_num)
    print("process", path, minv, maxv, pivot_num)
    save_data(save_path, [minv, maxv, pivot_num])
 


if __name__ == "__main__":
    pool = Pool(processes=10)
    path = "max_activations"
    save_path = "max_activations_res"
    layers = os.listdir(path)
    import time
    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    save_path_dir = os.path.join(save_path, current_time)
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir) 
    for i in range(len(layers)):
        layers[i] = (os.path.join(path, layers[i]), os.path.join(save_path_dir, layers[i]+"_max_act"))
    pool.map(find_max_act, layers)
    pool.close()
    pool.join()

    # bug = []
    # for i in range(len(layers)):
    #     layers[i] = os.path.join(path, layers[i])
    #     print(layers[i], os.path.exists(os.path.join(layers[i], "max_act_0.999")))
    #     if os.path.exists(os.path.join(layers[i], "max_act_0.999"))==False:
    #         bug.append(layers[i])
        # file_path = os.path.join(layers[i], "max_act")
        # with open(file_path, 'rb') as f:
        #     print(file_path, pickle.load(f))
    # pool.map(find_max_act, bug)

    # from backbone.MoSliceNet import *
    # from backbone.MoSliceNet_NoBN import moslicenetv10_nobn
    # from convert import mix_bn_to_conv, normalise_module
    # import torch
    # save_path = "max_activations_res/2020-07-21T14:59"
    # model = moslicenetv10_nobn()
    # pretrained = "checkpoint/2020-07-11T19:27moslicenet_relu_128/epoch_65_33.pth.tar" #"checkpoint/2020-07-04T23:11moslicenetv10_4_add_relu_resume/epoch_59_26.pth.tar"
    # name_map = [("blocks.0.0.conv_dw","blocks.0.0.bn1"), ("blocks.0.0.conv_pw","blocks.0.0.bn2"), ("conv_stem", "bn1"), ("conv_head", "bn2"),
    #     ("conv_pwl","bn3"), ("conv_pw","bn1"), ("conv_dw", "bn2")]
    # mix_bn_to_conv(model, pretrained, conv_names=["conv"], bn_names=["bn"], fc_names=["classifier"], name_map=name_map)
    # # print(normalise(model, path))
    # compensation = 1.01 #(300/230)^(1/80)1.0#
    # max_act = normalise_max(model, save_path)
    # for key in max_act:
    #     max_act[key] = 1.0
    # normalise_module(model, "", max_act, 1.0, compensation)
    # # print(model.blocks[5][2].scale.scale)
    # torch.save({'state_dict': model.state_dict()}, "checkpoint/0_pretrained_128/moslicenet_relu_128_epoch_65_33_nobn_compensation.pth.tar")

    
    
    

