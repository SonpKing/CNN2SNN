import pickle
import math
import time
# import node_alloc as alloc
from mapping import gen_input as gen_in
import os
from . import transmitter2 as ts
import numpy as np
from PIL import Image
from time import sleep

def read_vt(x,y ,neu_list):
    xy = "%03x" % (( x<< 6 )+ y)
    f = open("config/read.txt", "w")
    for i in neu_list:
        ins = "5E124"+ xy + "C71\n5C180000060\n"
        f.write(ins)
        neural_id = "%02x" % (i)
        ins = "5D1000001" + neural_id + "\n"
        f.write(ins)
        ins = "5E124"+ xy + "c71\n5D100000060\n" 
        f.write(ins)
    f.close()  

def read_image(file_path, size=32):
    img_obj = Image.open(file_path)
    img_obj = img_obj.resize((size, size), Image.BILINEAR)
    img_array = np.array(img_obj, dtype=np.uint8).astype(np.float)
    img_array = img_array / 255.0
    img_array = img_array.ravel()
    print(img_array.shape)
    return img_array

def read_input_config():
    with open('config/pickle/layerWidth1_1', 'rb') as f:
        layerWidth = pickle.load(f)
        # print(layerWidth)
    with open('config/pickle/nodelist1_1', 'rb') as f:
        nodelist = pickle.load(f)

    with open('connections/start_to_input_chip0', 'rb') as f:
        in_conv1 = pickle.load(f)

    print("load config over")   
    return layerWidth, nodelist, in_conv1


def mfc_to_com(image, in_head, layerWidth, nodelist, in_conv1, vth=100):
    spiketrain = []
    for i in range(3072):
        temp = [i, [1]]
        spiketrain.append(temp)   
    new_con = in_conv1
    new_con[:, 2] = np.rint((np.array(image) * vth))

    input_node_map = {}
    neuron_num = int(math.ceil(layerWidth[1] / float(len(nodelist[0]))))

    for line in new_con:
        dst = int(line[1])
        node_x = nodelist[0][dst // neuron_num][0]
        node_y = nodelist[0][dst // neuron_num][1]
        nodenumber = node_x * 64 + node_y
        if not nodenumber in input_node_map.keys():
            input_node_map[nodenumber] = {}
        input_node_map[nodenumber].update({dst % neuron_num: dst})
    gen_in.change_format(new_con)

    inputlist, rowlist = gen_in.gen_inputdata_list(new_con, spiketrain, input_node_map, int(1), in_head)
    # if(in_head == "40000") :
    #     gen_in.gen_inputdata(new_con, self.spiketrain, input_node_map, int(1), in_head, "input1.txt", "row.txt")
    # elif (in_head == "80000"):
    #     gen_in.gen_inputdata(new_con, self.spiketrain, input_node_map, int(1), in_head, "input2.txt", "row.txt")
    # else:
    #     gen_in.gen_inputdata(new_con, self.spiketrain, input_node_map, int(1), in_head, "input3.txt", "row.txt")
    

    return inputlist, rowlist

def generate_multi_input(image1, image2, image3):
    layerWidth, nodelist, in_conv1 = read_input_config()
    inputlist1, rowlist = mfc_to_com(image1, "40000", layerWidth, nodelist, in_conv1)
    inputlist2, rowlist = mfc_to_com(image2, "80000", layerWidth, nodelist, in_conv1)
    inputlist3, rowlist = mfc_to_com(image3, "C0000", layerWidth, nodelist, in_conv1)
    return inputlist1 + inputlist2 + inputlist3, rowlist

def fit_input(inputs):
    if inputs.shape[-1] == 3:
        pass
    elif inputs.shape[0] == 3:
        inputs = inputs.transpose((1, 2, 0))
    else:
        print("input shape error", inputs.shape)
    return inputs.ravel()

class DarwinDev:
    def __init__(self, IP, port, tick_time, class_num=8):
        address = (IP, port)
        self.trans = ts.Transmitter()
        self.trans.connect_lwip(address)

        print("tcp connect succeed", IP)
        # self.trans.asic_reset()
        self.trans.set_tick_time(tick_time)
        
        self.trans.send_config("config/1_1config.txt")
        print('configA send done', IP)
        self.trans.send_config("config/1_2config.txt")
        print('configB send done', IP)
        self.trans.send_config("config/1_3config.txt")
        print('configC send done', IP)
        self.class_num = class_num
        
    def get_result(self):
        return self.spikes_all

    def reset(self):
        print("send clear")
        self.trans.send_clear("config/1_1clear.txt")
        print("send clear done")
        self.trans.send_config("config/1_1enable.txt")
        print("send enable done")
    
    def eliminate(self , chip_order):
        self.trans.asic_reset()
        if(chip_order == 'A'):
            self.trans.send_config("config/re_config_A.txt")
        elif(chip_order == 'B'):
            self.trans.send_config("config/re_config_B.txt")
        elif(chip_order == 'C'):
            self.trans.send_config("config/re_config_C.txt")
        else:
            
            self.trans.send_config("config/re_config_A.txt")
            self.trans.send_config("config/re_config_B.txt")
            self.trans.send_config("config/re_config_C.txt")
        

    def run(self, inputlist, rowlist, ticks=105, show=False):
        ss_time = time.time()
        self.spikes_all = np.zeros(42) 
        self.trans.send_input_3chip_list(inputlist, rowlist, self.class_num) 
        for i in range(ticks):
            # print ('======tick=========  ' + str(i))
            spikes = self.trans.auto_tick(len(inputlist), self.class_num)
            
            # spikes = trans.send_input('input.txt', 'row.txt')
            spikes = np.asarray(spikes)
            self.spikes_all += spikes
            if show:
                self.trans.send_read_ins("config/read.txt")
        print("spike_time: ", time.time() - ss_time)
                

        # for i in range(20):
        #     self.trans.tick_alone() 
        # self.reset() 


def read_connections(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    data = sorted(data, key=lambda x: x[0]) 
    data = [item.tolist() for item in data]
    return data 


if __name__ == "__main__":
    # ------------------------------------- input ----------------------------------------------
    darwin = DarwinDev("192.168.1.10", 7, 2200000, "1_1config.txt")
    path = "data/18.jpg"
    data = read_image(path)

    print(darwin.run(data))

