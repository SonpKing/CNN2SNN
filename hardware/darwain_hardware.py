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

class DarwinDev:
    def __init__(self, IP, port, tick_time, class_num=8):
        address = (IP, port)
        self.trans = ts.Transmitter()
        self.trans.connect_lwip(address)

        print("tcp connect succeed")

        self.trans.set_tick_time(tick_time)



        self.trans.send_config("config/1_1config.txt")
        print('configA send done')
        self.trans.send_config("config/1_2config.txt")
        print('configB send done')
        self.trans.send_config("config/1_3config.txt")
        print('configC send done')
        
        self.read_input_config()

        self.class_num = class_num


    def run(self, image1, image2, image3, ticks=105, show=False):
        inputlist1, rowlist = self.mfc_to_com(image1, "40000")
        inputlist2, rowlist = self.mfc_to_com(image2, "80000")
        inputlist3, rowlist = self.mfc_to_com(image3, "C0000")
        # read_vt(1, 1, list(range(8)))
        ss_time = time.time()
        self.spikes_all = np.zeros(45) 
        self.trans.send_input_3chip_list(inputlist1, inputlist2, inputlist3, rowlist) 
        for i in range(ticks):
            if show: 
                print ('======tick=========  ' + str(i))
            spikes = self.trans.auto_tick(len(inputlist1*3))
            
            # spikes = trans.send_input('input.txt', 'row.txt')
            spikes = np.asarray(spikes)
            self.spikes_all += spikes
            if show:
                self.trans.send_read_ins("config/read.txt")
        print("spike_time: ", time.time() - ss_time)
                

        # for i in range(20):
        #     self.trans.tick_alone() 
        # self.reset() 
        
    def get_result(self):
        
        return self.spikes_all[:self.class_num]

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
            

    def fit_input(self, inputs):
        if inputs.shape[-1] == 3:
            pass
        elif inputs.shape[0] == 3:
            inputs = inputs.transpose((1, 2, 0))
        else:
            print("input shape error", inputs.shape)
        return inputs.ravel()

    def mfc_to_com(self, image, in_head, vth=100 ):
        new_con = self.in_conv1
        new_con[:, 2] = np.rint((np.array(image) * vth))

        input_node_map = {}
        neuron_num = int(math.ceil(self.layerWidth[1] / float(len(self.nodelist[0]))))

        for line in new_con:
            dst = int(line[1])
            node_x = self.nodelist[0][dst // neuron_num][0]
            node_y = self.nodelist[0][dst // neuron_num][1]
            nodenumber = node_x * 64 + node_y
            if not nodenumber in input_node_map.keys():
                input_node_map[nodenumber] = {}
            input_node_map[nodenumber].update({dst % neuron_num: dst})
        gen_in.change_format(new_con)

        inputlist, rowlist = gen_in.gen_inputdata_list(new_con, self.spiketrain, input_node_map, int(1), in_head)
        # if(in_head == "40000") :
        #     gen_in.gen_inputdata(new_con, self.spiketrain, input_node_map, int(1), in_head, "input1.txt", "row.txt")
        # elif (in_head == "80000"):
        #     gen_in.gen_inputdata(new_con, self.spiketrain, input_node_map, int(1), in_head, "input2.txt", "row.txt")
        # else:
        #     gen_in.gen_inputdata(new_con, self.spiketrain, input_node_map, int(1), in_head, "input3.txt", "row.txt")
        

        return inputlist, rowlist


    def read_input_config(self):
        with open('config/pickle/connfiles1_1', 'rb') as f:
            self.connfiles = pickle.load(f)

        with open('config/pickle/layerWidth1_1', 'rb') as f:
            self.layerWidth = pickle.load(f)
            # print(layerWidth)


        with open('config/pickle/nodelist1_1', 'rb') as f:
            self.nodelist = pickle.load(f)
            # print(nodelist)


        with open('connections/start_to_input_chip0', 'rb') as f:
            self.in_conv1 = pickle.load(f)

        print("load config over")

        self.spiketrain = []
        for i in range(3072):
            temp = [i, [1]]
            self.spiketrain.append(temp)      


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

