from hardware import DarwinDev
from hardware.create_conf import create_config
from hardware.darwain_hardware import read_connections
from hardware.insert_zeros import insert_zeros
import numpy as np 
from time import sleep
from util import data_loader

def sim():
    _, val_loader = data_loader("C:\\Users\\dell\\Desktop\\Detect_Data", batch_size=1, img_size=32, workers=1, dataset="imagenet") 
    ticks = 120
    total_acc  = 0
    sim = DarwinDev("192.168.1.10", 7, 220000, "1_1config.txt")
    for _ in range(10):
        for it, (inputs, targets) in enumerate(val_loader):
            # images = np.zeros((32, 32, 3))
            # images = sim.fit_input(images)
            # sim.run(images, 25, show=False)

            inputs = inputs.numpy()[0]
            # inputs = np.zeros((32, 32, 3))
            sim.reset()
            # sleep(1.0)
            images = sim.fit_input(inputs)
            sim.run(images, ticks, show=False)

            sim_res = sim.get_result()
            print(sim_res)
            sim_res = np.argmax(sim_res)
            if sim_res == targets[0]:
                total_acc += 1
            print(it,sim_res, targets[0], total_acc/(it+1)*100)
            
            # sleep(1.0)

if __name__ == "__main__":
    # create_config("connections_banana_100_7", 100, 1)
    sim()
    # data = read_connections('connections_banana_100_7\\net.blocks.1.pool_to_net.blocks.2.conv_chip0')[1000:1200]
    # for d in data:
    #     print(d)

    # insert_zeros("connections_banana_100", 40, "connections_banana_100_7")