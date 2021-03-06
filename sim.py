from util import data_loader
from simulation.nest_sim import Simulator
import numpy as np
def main():
    _, val_loader = data_loader("/home/jinxb/Project/data/Detect_Annoed", batch_size=1, img_size=32, workers=1, dataset="imagenet")
    sim = Simulator(scale=60, reset_sub=True)
    sim.create_net("connections/", "input", "net.classifier")
    ticks = 200
    total_acc  = 0
    max_conf = 0
    for it, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.numpy()[0]
        sim.reset()
        sim.fit_input(inputs)
        sim.run(ticks)
        sim_res = sim.get_result()
        max_conf = max(max_conf, sim_res[-1])
        print(sim_res)
        sim_res = np.argmax(sim_res)
        if sim_res == targets[0]:
            total_acc += 1
        print(it,sim_res, targets[0], total_acc/(it+1)*100)
        # input()
    print(max_conf)

main()