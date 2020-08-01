import pickle
import math
import time
# import node_alloc as alloc
from mapping import gen_input as gen_in
import os
import transmitter2 as ts
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
    img_array = img_array.transpose((2, 0, 1)).ravel()
    print(img_array.shape)
    return img_array

class DarwinDev:
    def __init__(self, IP, port, tick_time, config_file, class_num=8):
        address = (IP, port)
        self.trans = ts.Transmitter()
        self.trans.connect_lwip(address)

        print("tcp connect succeed")

        self.trans.set_tick_time(tick_time)

        self.trans.send_config(os.path.join("config", config_file))
        print('config send done')
        
        self.read_input_config()

        self.class_num = class_num


    def run(self, image, ticks=105, show=False):
        inputlist, rowlist = self.mfc_to_com(image)
        # read_vt(1, 1, list(range(8)))
        self.spikes_all = np.zeros(10)
        for i in range(ticks):
            if show: 
                print ('======tick=========  ' + str(i))
            spikes = self.trans.send_input_list(inputlist, rowlist)
            # spikes = trans.send_input('input.txt', 'row.txt')
            spikes = np.asarray(spikes)
            self.spikes_all += spikes
            if show:
                self.trans.send_read_ins("config/read.txt")

        # for i in range(10):
        #     self.trans.tick_alone(show) 
        # self.reset() 
        
    def get_result(self):
        return self.spikes_all[:self.class_num]

    def reset(self):
        print("send clear")
        self.trans.send_clear("config/1_1clear.txt")
        print("send clear done")
        self.trans.send_config("config/1_1enable.txt")
        print("send enable done")

    def fit_input(self, inputs):
        return inputs.transpose((1, 2, 0)).ravel()

    def mfc_to_com(self, image, vth=100):
        new_con = self.in_conv1
        new_con[:, 2] = np.rint((np.array(image) * vth))

        input_node_map = {}
        neuron_num = int(math.ceil(self.layerWidth[1] / float(len(self.nodelist[0]))))

        for line in new_con:
            dst = int(line[1])
            node_x = self.nodelist[0][dst // neuron_num][0]
            node_y = self.nodelist[0][dst // neuron_num][1]
            nodenumber = node_x * 24 + node_y
            if not nodenumber in input_node_map.keys():
                input_node_map[nodenumber] = {}
            input_node_map[nodenumber].update({dst % neuron_num: dst})
        gen_in.change_format(new_con)

        inputlist, rowlist = gen_in.gen_inputdata_list(new_con, self.spiketrain, input_node_map, int(1))

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


if __name__ == "__main__":
    # ------------------------------------- input ----------------------------------------------
    darwin = DarwinDev("192.168.1.10", 7, 2200000, "1_1config.txt")
    path = "data/18.jpg"
    data = read_image(path)

    print(darwin.run(data))

