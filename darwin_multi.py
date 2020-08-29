from hardware import DarwinDev
# from hardware.create_conf_nice5 import create_config
from hardware.create_conf_anno_loss_weight_7 import create_config
from hardware.darwain_hardware import read_image, read_connections, fit_input, generate_multi_input, edit_connections
import numpy as np 
from time import sleep
from util import data_loader
import time
from hardware.insert_zeros import insert_zeros


def sim():
    batch_size = 3
    # _, val_loader = data_loader("C:\\Users\\zjlab\\Desktop\\Detect_Data_anno\\Detect_Data", batch_size=3, img_size=32, workers=1, dataset="imagenet") 
    _, val_loader = data_loader("C:\\Users\\zjlab\\Desktop\\Detect_Annoed", batch_size=3, img_size=32, workers=1, dataset="imagenet")#
    ticks = 200
    total_acc  = 0
    class_num = 15
    sim = DarwinDev("192.168.1.59", 7, 220000, class_num)
    
    # for _ in range(1):
    # for idx in range(10):
    max_conf = 0
    for it, (inputs, targets) in enumerate(val_loader):
        # inputs = read_image("data/86.jpg")
        shows = [False]*100
        input1 = inputs.numpy()[0]
        image1 = fit_input(input1)
        input2 = inputs.numpy()[1]
        image2 = fit_input(input2)
        input3 = inputs.numpy()[2]
        image3 = fit_input(input3)

        # print(input1==input2)

        sim.eliminate('all')
        print("eliminate")
        s_time = time.time()
        inputlist, rowlist = generate_multi_input(image1, image2, image3)
        sim.run(inputlist, rowlist, class_num*3, ticks, show=shows[it] )
        print("hardware time:", time.time() - s_time)

        # sim.reset()
        # TODO： 没有 insertzeros
        
        
        
        
        
        sim_res = sim.get_result()
        print(sim_res[0:class_num])
        print(sim_res[class_num:class_num*2])
        print(sim_res[class_num*2:])
        max_conf = max(max_conf, sim_res[class_num-1])
        max_conf = max(max_conf, sim_res[2*class_num-1])
        max_conf = max(max_conf, sim_res[3*class_num-1])

        sim_res1 = np.argmax(sim_res[0:class_num])
        sim_res2 = np.argmax(sim_res[class_num:class_num*2])
        sim_res3 = np.argmax(sim_res[class_num*2:])
        
        if sim_res1 == targets[0]:
            total_acc += 1
        if sim_res2 == targets[1]:
            total_acc += 1
        if sim_res3 == targets[2]:
            total_acc += 1
        print(it, total_acc/(it*batch_size+1)*100)
    print(max_conf)



            
            

        




if __name__ == "__main__":
    # create_config()
    sim()
    # data = read_connections('connections_anno_7\\net.global_pool_to_net.classifier_chip0')[0:500]
    # for d in data:
    #     print(d)
    # edit_connections("connections_anno_loss_weight_7\\net.global_pool_to_net.classifier_chip0", 14, 4)
    # insert_zeros("connections_anno_loss_weight",40,'connections_anno_loss_weight_7')
