import nest
import numpy as np
import os, pickle
from convert import parse_name
from PIL import Image


class Simulator:
    def __init__(self, scale, vth = 100, merge_layer="", merge_num=0, threads=1, chips=1, reset_sub=True):
        super().__init__()
        #every thread can only own 2^19 connections, or raise >max_lid error
        self.scale = scale * 1.0
        self.input_vth = vth * 1.0
        nest.SetKernelStatus({"local_num_threads": threads})
        nest.SetKernelStatus({'resolution':1.0})
        nest.CopyModel("iaf_delta_noleak", "res_iaf", params={'t_ref': 0.0, "I_e":0.0, "V_th":np.around(self.scale), "V_m":0.0, 'E_L': 0.0, 'V_reset':0.0, 'subtraction':reset_sub}) 
        nest.CopyModel("iaf_delta_noleak", "res_iaf_input", params={'t_ref': 0.0, "I_e":0.0, "V_th":self.input_vth, "V_m":0.0, 'E_L': 0.0, 'V_reset':0.0, 'subtraction':True}) 
        nest.CopyModel("iaf_delta_noleak", "spike_iaf", params={'t_ref': 0.0, "I_e":1.0, "V_th":1.0, "V_m":0.0, 'E_L': 0.0, 'V_reset':0.0, 'subtraction':reset_sub}) 
        self.neuron = "res_iaf"
        self.neuron_i = "res_iaf_input"
        self.neuron_c = "spike_iaf"
        self.layers_num = dict()
        self.merge_layer = merge_layer
        self.merge_neurons_num = merge_num
        self.layers = dict()
        self.chips = chips


    def create_layer(self, layer, neuron_type=None):
        if layer not in self.layers:
            assert layer in self.layers_num
            if neuron_type == None:
                neuron_type = self.neuron
            neurons = nest.Create(neuron_type, self.layers_num[layer])
            self.layers[layer] = neurons[0]
            print("creating layer", layer, "start from", neurons[0], ", total", self.layers_num[layer], "neurons")

    def create_layer_bias(self, layer, delay):
        if layer not in self.layers:
            assert layer in self.layers_num
            neuron_dict = {"V_m": -1.0 * nest.GetDefaults(self.neuron_c, "I_e") * delay}
            neurons = nest.Create(self.neuron_c, self.layers_num[layer], neuron_dict)
            self.layers[layer] = neurons[0]
            print("creating layer", layer, "start from", neurons[0], ", total", self.layers_num[layer], "neurons")


    def connect(self, layer1, layer2, conns):
        
        if layer2[:-1] == self.merge_layer:
            print("merge", layer2, "to", layer2[:-1]+'0')
            layer2 = layer2[:-1]+'0'
        if layer1 not in self.layers_num or layer2 not in self.layers_num:
            print("please count neurons")
            return
        if layer1 not in self.layers:
            if "bias" in layer1:
                self.create_layer_bias(layer1, 0)
            else:
                self.create_layer(layer1)
        if layer2 not in self.layers:
            self.create_layer(layer2)
        pre_neurons = (np.array(conns[:, 0]) + self.layers[layer1]).tolist()
        post_neurons = (np.array(conns[:, 1]) + self.layers[layer2]).tolist()
        weights = conns[:, 2]
        nest.Connect(pre_neurons, post_neurons, "one_to_one", syn_spec={"weight":weights})
        print("connect", layer1, pre_neurons[0], "to", layer2, post_neurons[0], ", total", len(conns), "connections")


    def count_layer_neurons(self, layer1, layer2, conns):
        pre_neurons = conns[:, 0]
        post_neurons = conns[:, 1]
        tmp = list(post_neurons)
        tmp.sort()
        print("stastic", layer2, tmp[0], tmp[-1])
        if layer1 not in self.layers_num:
            self.layers_num[layer1] = max(pre_neurons) + 1
        else:
            # print(layer1, self.layers_num[layer1] <  max(pre_neurons) + 1)
            self.layers_num[layer1] = max(self.layers_num[layer1], max(pre_neurons) + 1)
        
        if layer2[:-1] == self.merge_layer:
            print("merge", layer2, "to", layer2[:-1]+'0')
            layer2 = layer2[:-1]+'0'
            if layer2 not in self.layers_num:
                self.layers_num[layer2] = self.merge_neurons_num
        else:
            if layer2 not in self.layers_num:
                self.layers_num[layer2] = max(post_neurons) + 1
            else:
                # print(layer2, self.layers_num[layer2] <  max(post_neurons) + 1)
                self.layers_num[layer2] = max(self.layers_num[layer2], max(post_neurons) + 1)

    def create_net(self, path, input_layer_name, output_layer_name):
        files = os.listdir(path)
        print("total", len(files), "files")
        for file in files:
            name_pre, name_post = parse_name(file)
            with open(path+file, "rb") as f:
                print(path+file)
                data = pickle.load(f)
            self.count_layer_neurons(name_pre, name_post, data)
            # print(name_pre, self.layers_num[name_pre], name_post, self.layers_num[name_post])
        print(self.layers_num)
        self.generate_input(input_layer_name)
        for file in files:
            name_pre, name_post = parse_name(file)
            with open(path+file, "rb") as f:
                data = pickle.load(f)
            self.connect(name_pre, name_post, data)
        self.detectors = self.add_detectors(output_layer_name)
        print("total connections:", len(nest.GetConnections()))


    def reset(self):
        nest.ResetNetwork()

    def add_detectors(self, output_layer_name):
        output_layer =  output_layer_name + '0'
        assert output_layer in self.layers_num
        assert output_layer in self.layers
        output_neurons = (np.arange(self.layers_num[output_layer]) + self.layers[output_layer]).tolist()
        print("detect from", output_neurons[0], "to", output_neurons[-1])
        detectors = nest.Create("spike_detector", self.layers_num[output_layer], params={"withgid": True, "withtime": True})
        detectors_name = output_layer_name + "_dectector"
        assert detectors_name not in self.layers_num
        self.layers_num[detectors_name] = len(detectors)
        self.layers[detectors_name] = detectors[0]
        nest.Connect(output_neurons, detectors, "one_to_one")
        print("generate output detectors", len(detectors))
        return detectors
        

    def generate_input(self, input_layer_name):
        print("generate input connections")
        self.conn = []
        start_neuron = nest.Create(self.neuron_c, 1)
        self.layers_num["start0"] = len(start_neuron)
        self.layers["start0"] = start_neuron[0]
        for i in range(self.chips):
            input_layer = input_layer_name + str(i)
            print(input_layer, self.chips)
            assert input_layer in self.layers_num         
            if input_layer not in self.layers:
                self.create_layer(input_layer, self.neuron_i)
            input_neurons = (np.arange(self.layers_num[input_layer]) + self.layers[input_layer]).tolist()
            nest.Connect(start_neuron, input_neurons, syn_spec={"weight":[[1.0] for i in input_neurons]})
            conn = nest.GetConnections(source=start_neuron, target=input_neurons)
            self.conn.append(sorted(conn, key = lambda k: k[1]))# local_num_threads is not 1, the order of conn cann't be promised. sort by target_id

    def fit_input(self, inputs):
        assert np.max(inputs) <= 1.0 and np.min(inputs) >= 0.0
        _, height, width = inputs.shape
        slice = int(np.sqrt(self.chips))
        _height, _width = height // slice, width // slice
        inputs = np.round(inputs * self.input_vth)
        if inputs.shape[-1] == 3:
            pass
        elif inputs.shape[0] == 3:
            inputs = inputs.transpose((1, 2, 0))
        else:
            print("input shape error", inputs.shape)
        for i in range(slice):
            for j in range(slice):
                img = list(inputs[i*_height: (i+1)*_height, j*_width: (j+1)*_width, :].ravel())
                nest.SetStatus(self.conn[i*slice+j], [{"weight":I_e} for I_e in img])
 

    def run(self, times):
        nest.Simulate(times)

    def get_result(self, layer_name=None):
        if layer_name == None:
            detectors = self.detectors
        else:
            detectors_name = layer_name + "_dectector"
            assert detectors_name in self.layers
            detectors = (np.arange(self.layers_num[detectors_name]) + self.layers[detectors_name]).tolist()
        dSD = nest.GetStatus(detectors, keys="events")
        spike_nums = [len(dsd["times"]) for dsd in dSD]
        return spike_nums

    def get_result2(self, layer_name=None, save_path = None):
        if layer_name == None:
            detectors = self.detectors
        else:
            detectors_name = layer_name + "_dectector"
            assert detectors_name in self.layers
            detectors = (np.arange(self.layers_num[detectors_name]) + self.layers[detectors_name]).tolist()
        dSD = nest.GetStatus(detectors, keys="events")
        spike_nums = [dsd["times"] for dsd in dSD]
        if save_path:
            with open(save_path,"wb") as f:
                pickle.dump(spike_nums, f)
        return spike_nums


def show_pic(inputs):
    print(inputs.shape)
    img = inputs.transpose((1, 2, 0)) * 255
    print(img.shape)
    img = np.rint(img).astype(np.uint8)
    img = Image.fromarray(img)
    img.show()
    
def read_image(file_path, size=32):
    img_obj = Image.open(file_path)
    img_obj = img_obj.resize((size, size), Image.BILINEAR)
    img_array = np.array(img_obj, dtype=np.uint8).astype(np.float)
    img_array = img_array / 255.0
    img_array = img_array.transpose((2, 0, 1))
    return img_array

def test(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    data = sorted(data, key=lambda x: x[0])
    for i in range(100):
        print(data[i].tolist())
    exit(0)

if __name__ == "__main__":
    # test("connections/input_to_net.blocks.0.conv_chip0")
    # print(nest.GetKernelStatus())
    from util import data_loader
    _, val_loader = data_loader("/home/jinxb/Project/data/Darwin_data2", batch_size=1, img_size=32, workers=1, dataset="imagenet")
    sim = Simulator(scale=70, reset_sub=True)
    sim.create_net("connections/", "input", "net.classifier")
    tics = 200
    total_acc  = 0
    for it, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.numpy()[0]
        sim.reset()
        sim.fit_input(inputs)
        sim.run(tics)
        sim_res = sim.get_result()
        print(sim_res)
        sim_res = np.argmax(sim_res)
        if sim_res == targets[0]:
            total_acc += 1
        print(it,sim_res, targets[0], total_acc/(it+1)*100)
        input()

# if __name__ == "__main__":
#     scale = 100.0
#     sim = Simulator(reset_sub=True)
#     sim.create_net("connections/", "input", "net.fc")
#     tics = 100
#     total_acc  = 0
    
#     pic_path = "/home/jinxb/Project/data/Darwin_data2/val/Apple" #Aeroplane
#     files = os.listdir(pic_path)
#     for it, file in enumerate(files):
#         file = "18.jpg"
#         inputs = read_image(os.path.join(pic_path, file))
#         # show_pic(inputs)
#         targets = [1]
#         sim.reset()
#         sim.fit_input(inputs)
#         sim.run(tics)
#         sim_res = sim.get_result()
#         print(sim_res)
#         sim_res = np.argmax(sim_res)
#         if sim_res == targets[0]:
#             total_acc += 1
#         print(it,sim_res, targets[0], total_acc/(it+1)*100)


# if __name__ == "__main__":
#     # print(nest.GetKernelStatus())
#     scale = 35.0
#     sim = Simulator(reset_sub=False)
#     sim.create_net("connections_new_bak/", "input", "net.classifier")
#     tics = 10
#     pic_path = "/home/jinxb/Project/data/Darwin_data2/val/Apple/18.jpg" #Aeroplane
#     sim.add_detectors("net.conv_stem")
#     inputs = read_image(pic_path)
#     sim.reset()
#     sim.fit_input(inputs)
#     sim.run(tics)
#     sim_res = sim.get_result2("net.conv_stem", "spike_num")
#     print(sim_res)



# %%  
        # # inputs = np.arange(1, 32*32*3+1).reshape((3, 32, 32)).astype(np.float) * 0.001
        # # inputs = np.ones((3, 32, 32), dtype=np.float) * 0.01
        # # inputs[0, 0, 0] *= 10
        # # inputs[0, 1, 0] *= 20
        # # inputs[0, 2, 0] *= 30
        # # inputs[1, 0, 0] *= 40
        # sim.fit_input(inputs)
        # # print(nest.GetConnections(target=[2]))
        # sim.run(101)
        # sim_res =  sim.get_result("input")
        # # print(sim_res)
        # print(inputs[:, -4:, -4:])
        # print(np.array(sim_res).reshape(inputs.shape)[:, -4:, -4:])
        # print(np.sum((np.round(inputs*sim.input_vth)*100/sim.input_vth).astype(np.int)!=np.array(sim_res).reshape(inputs.shape)))
        # if input()=="c":
        #     exit(0)

    # sim.reset()
    # inputs = np.ones((32, 32, 3), dtype=np.float)
    # sim.fit_input(inputs)
    # print("start simulation")
    # sim.run(10)




# # import torch
# # from spikeresnet import If

# # net = If()
# # net.eval()
# # input = torch.ones(1,3,4,4)*0.5
# # for _ in range(10):
# #     out = net(input)
# # print(net.sum_spike)
# import nest
# nest.SetKernelStatus({'resolution':1.0})
# print(nest.GetKernelStatus())
# nest.CopyModel("iaf_delta_noleak","res_iaf", params={'t_ref': 0.0, "I_e":0.0, "V_th":1.0, "V_m":0.0, 'E_L': 0.0, 'V_reset':0.0, 'subtraction':True}) 
# # neuron_dict = {"I_e":50.0, "V_m":-960.0, "V_th":150.0}
# # test_iaf = nest.Create('res_iaf', 1, neuron_dict)
# # multimeter = nest.Create("multimeter", params= {"withtime":True, "record_from":["V_m"]})
# # nest.Connect(multimeter, test_iaf)
# # nest.Simulate(30)
# # dmm = nest.GetStatus(multimeter)[0]
# # Vms = dmm["events"]["V_m"]
# # ts = dmm["events"]["times"]
# # print(ts,Vms)
# # import pylab
# # pylab.figure(1)
# # pylab.plot(ts, Vms)
# # pylab.savefig("sim.png")

# # test_neuron = nest.Create('dc_generator')
# # out = nest.Create('dc_generator')
# # nest.Connect(test_neuron, out, "one_to_one", syn_spec={"weight":1.0})
# ins = nest.Create('res_iaf',2000)
# outs = nest.Create('res_iaf',2000)
# nest.Connect(ins, outs)
# print(len(nest.GetConnections()))
# print("--------")
# nest.Disconnect(ins, outs, "one_to_one")
# print(len(nest.GetConnections()))
