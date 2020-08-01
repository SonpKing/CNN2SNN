from brian2 import *

import pickle
import os


def parse_name(file_name):
    layer1, layer2 = file_name.split("_to_")
    chip = layer2[-1]
    layer2 = layer2[:-6]
    if layer2[-4:] == "bias":
        layer1 += "_bias"
        layer2 = layer2[:-5]
    return (layer1+chip, layer2+chip)

def create_net():
    path = "connections3/"
    files = os.listdir(path)
    print("total", len(files), "files")
    sim = Simulator("net.blocks.5.0.conv_pwl", 192*4*4)
    for file in files:
        name_pre, name_post = parse_name(file)
        with open(path+file, "rb") as f:
            data = pickle.load(f)
        sim.count_all_neurons(name_pre, name_post, data)
    for file in files:
        name_pre, name_post = parse_name(file)
        with open(path+file, "rb") as f:
            data = pickle.load(f)
        sim.connect(name_pre, name_post, data)
    return sim

class Simulator:
    def __init__(self, merge_layer="", merge_num=0):
        super().__init__()
        self.layers = dict()
        self.eqs = 'dv/dt = 0 / ms : 1'
        self.vth = 1.0
        self.net = Network(collect())
        self.has_stored = False
        self.merge_layer = merge_layer
        self.merge_neurons_num = merge_num
        self.layers_num = dict()


    def count_all_neurons(self, layer1, layer2, conns):
        pre_neurons = conns[:, 0]
        post_neurons = conns[:, 1]
        if layer1 not in self.layers:
            self.layers_num[layer1] = max(pre_neurons) + 1
        else:
            self.layers_num[layer1] = max(self.layers_num[layer1], max(pre_neurons) + 1)
        
        if layer2[:-1] == self.merge_layer:
            print("merge", layer2, "to", layer2[:-1]+'0')
            layer2 = layer2[:-1]+'0'
            if layer2 not in self.layers:
                self.layers_num[layer2] = self.merge_neurons_num
        else:
            if layer2 not in self.layers:
                self.layers_num[layer2] = max(post_neurons) + 1
            else:
                self.layers_num[layer2] = max(self.layers_num[layer2], max(post_neurons) + 1)


    def connect(self, layer1, layer2, conns):
        print("connect", layer1, "to", layer2, ", total", len(conns), "connections")
        if layer2[:-1] == self.merge_layer:
            print("merge", layer2, "to", layer2[:-1]+'0')
            layer2 = layer2[:-1]+'0'
        if layer1 not in self.layers_num or layer2 not in self.layers_num:
            print("please count neurons")
            return
        if layer1 not in self.layers:
            self.create_layer(layer1, self.layers_num[layer1])
        if layer2 not in self.layers:
            self.create_layer(layer2, self.layers_num[layer2])
        pre_neurons = conns[:, 0]
        post_neurons = conns[:, 1]
        S = Synapses(self.layers[layer1], self.layers[layer2], 'w : 1', on_pre='v_post += w', delay=0.0*ms)
        S.connect(i=pre_neurons, j=post_neurons)
        S.w = conns[:, 2]
        self.net.add(S)
        

    def create_layer(self, layer_name, neurons_nums):
        if layer_name not in self.layers:
            vth = self.vth
            print("creating layer", layer_name, ", total", neurons_nums, "neurons")
            self.layers[layer_name] = NeuronGroup(neurons_nums, self.eqs, threshold='v >= '+str(vth), reset='v -= '+str(vth), method='euler')
            self.net.add(self.layers[layer_name])
            

    def store(self):
        print("store the simulator status")
        self.net.store('initialized')

    def restore(self):
        print("restore the simulator status")
        self.net.restore('initialized')

    def run(self, seconds):
        # if not self.has_stored:
        #     self.store()
        #     self.has_stored = True
        # else:
        #     self.restore()
        self.net.run(seconds*ms)


if __name__ == "__main__":
    start_scope()
    set_device('cpp_standalone')
    # prefs.devices.cpp_standalone.openmp_threads = 4
    defaultclock.dt = 1*ms

    sim = create_net()
    sim.run(2)
    # input_neurons = NeuronGroup(1, 'dv/dt = 1 / ms : 1', threshold='v >= 1', reset='v -= 1', method='euler')
    # output_neurons = NeuronGroup(1, 'dv/dt = 0 / ms : 1', threshold='v >= 1', reset='v -= 1', method='euler')
    # S = Synapses(input_neurons, output_neurons, 'w : 1', on_pre='v_post += w', delay=0.0*ms)
    # S.connect(i=[0], j=[0])
    # S.w = [2]
    # monitor1 = SpikeMonitor(input_neurons)
    # monitor2 = SpikeMonitor(output_neurons)
    # M_in = StateMonitor(input_neurons, 'v', record=True)
    # M_out = StateMonitor(output_neurons, 'v', record=True)
    # run(10*ms)
    # print(monitor1.spike_trains())
    # print(monitor2.spike_trains())
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(M_in.t/ms, M_in.v[0])
    # ax[1].plot(M_out.t/ms, M_out.v[0])
    # ax[0].set_xlabel('time')
    # ax[0].set_ylabel('v')
    # ax[1].set_xlabel('time')
    # ax[1].set_ylabel('v')
    # plt.show()



# start_scope()
# set_device('cpp_standalone')
# prefs.devices.cpp_standalone.openmp_threads = 4
# defaultclock.dt = 1*ms
# input_neurons = NeuronGroup(np.array(inputs.shape).prod(), dc, threshold='v >= 5', reset='v -= 5', method='euler')
# bias_neurons = NeuronGroup(outputs.shape[0], dc, threshold='v >= 5', reset='v -= 5', method='euler')
# output_neurons = NeuronGroup(np.array(outputs.shape).prod(), eqs, threshold='v >= 200', reset='v -= 200', method='euler')
# S = Synapses(input_neurons, output_neurons, 'w : 1', on_pre='v_post += w', delay=0.0*ms)
# weight_conns = np.array(conns[0])
# S.connect(i=weight_conns[:, 0].astype(int).tolist(), j=weight_conns[:, 1].astype(int).tolist())
# S.w = weight_conns[:, 2]