from hardware import DarwinDev, create_config
import numpy as np 
from time import sleep
from util import data_loader

def sim():
    _, val_loader = data_loader("C:/Users/dell/Desktop/Darwin_data2", batch_size=1, img_size=32, workers=1, dataset="imagenet") 
    ticks = 100
    total_acc  = 0
    sim = DarwinDev("192.168.1.10", 7, 220000, "1_1config.txt")
    for it, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.numpy()[0]
        sim.reset()
        # sleep(1.0)
        images = sim.fit_input(inputs)
        sim.run(images, ticks)
        sim_res = sim.get_result()
        print(sim_res)
        sim_res = np.argmax(sim_res)
        if sim_res == targets[0]:
            total_acc += 1
        print(it,sim_res, targets[0], total_acc/(it+1)*100)
        # sleep(1.0)

if __name__ == "__main__":
    # create_config()
    sim()