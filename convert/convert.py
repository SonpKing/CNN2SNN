'''
@author: pking
@email: pking96@163.com

'''

import numpy as np 
import torch
import torch.nn as nn

from util import get_state, load_pruned

import math
import pickle
import os

def is_contain_name(layer_name, part_name_list, sub_name=""):
    for part_name in part_name_list:
        if part_name in layer_name and sub_name in layer_name:
            return True
    return False

def error_report(info):
    print(info)
    exit(0)


# def save_max_activations(net, file_name, path="max_activations"):
#     res = dict()
#     for name, module in net.named_modules():
#         if isinstance(module, MyRelu):
#             max_act = module.get_max_act()
#             max_act.sort()
#             res[name] = [max_act[0], max_act[int(len(max_act)*0.9)], max_act[-1]]
#     if not os.path.exists(path):
#         os.mkdir(path)
#     with open(os.path.join(path, file_name), 'wb') as f:
#         pickle.dump(res, f)


def parse_name(file_name):
    layer1, layer2 = file_name.split("_to_")
    chip = layer2[-1]
    layer2 = layer2[:-6]
    if layer2[-4:] == "bias":
        layer1 += "_bias"
        layer2 = layer2[:-5]
    return (layer1+chip, layer2+chip)


def get_network_order(path):
    files = os.listdir(path)
    print("total", len(files), "files")
    layer_file_map = dict()
    for file in files:
        layer1, _ = parse_name(file)
        layer_file_map[layer1] = file
    return layer_file_map
    

def mute_prune_connections(path, save_path):
    files = os.listdir(path)
    print("total", len(files), "files")
    layer_neurons = dict()
    layer_file_map = get_network_order(path)
    ind_layer = "input0"
    with open(os.path.join(path, layer_file_map[ind_layer]), "rb") as f:
        conns = pickle.load(f)
    layer_neurons[ind_layer] = set(conns[:, 0])
    total_neurons = [max(conns[:,0])+1]
    while ind_layer in layer_file_map:
        file = layer_file_map[ind_layer]
        layer1, layer2 = parse_name(file)
        ind_layer = layer2
        with open(os.path.join(path, file), "rb") as f:
            conns = pickle.load(f)
            total_neurons.append(max(conns[:,1])+1)
        layer_neurons[layer1] &= set(conns[:, 0])
        tmp_conn = []
        for conn in conns:
            if conn[0] in layer_neurons[layer1]:
                tmp_conn.append(conn)
        tmp_conn = np.asarray(tmp_conn)
        layer_neurons[layer2] = set(tmp_conn[:, 1])
        print(layer1, len(layer_neurons[layer1]))
        print(layer2, len(layer_neurons[layer2]))
    print(total_neurons)

    layer_shuffle = dict()
    for layer in layer_neurons:
        all_neurons = list(layer_neurons[layer])
        all_neurons.sort()
        print(layer, len(all_neurons))
        layer_shuffle[layer] = dict()
        for i in range(len(all_neurons)):
            layer_shuffle[layer][all_neurons[i]] = i
        print(list(layer_shuffle[layer].items())[:10])

    for file in files:
        layer1, layer2 = parse_name(file)
        with open(os.path.join(path, file), "rb") as f:
            conns = pickle.load(f)
        new_conns = []
        layer1_shuffle = layer_shuffle[layer1]
        layer2_shuffle = layer_shuffle[layer2]
        for conn in conns:
            if conn[0] in layer1_shuffle and conn[1] in layer2_shuffle:
                new_conns.append((layer1_shuffle[conn[0]], layer2_shuffle[conn[1]], *conn[2:]))
        print(file, len(conns), len(new_conns), new_conns[0])
        save_connections(new_conns, file, save_path)
    
    for layer in layer_shuffle:
        if "input" in layer:
            conns = []
            for nid in layer_shuffle[layer].values():
                conns.append((nid, nid, 1.0, 0))
            print("generate", len(conns), "forward connections")
            save_connections(conns, "start_to_input_chip0", save_path)
            break

        


def convert_prune_weight(state):
    new_state = dict()
    for key in state:
        if "orig" in key:
            key2 = key.replace("orig", "mask")
            new_state[key[:-5]] = (state[key] * state[key2])
        elif "mask" not in key:
            new_state[key] = state[key]
    return new_state


def get_bn_name(conv_name, name_map):
    for (old, new) in name_map:
        if old in conv_name:
            return conv_name.replace(old, new)
    return None


def mix_bn_to_conv(net, state_path, conv_names=["conv"], bn_names=["bn"], fc_names=["fc"], name_map={}, device=None):
    '''
    Combine the batch_normal layer with the convolutional layer.

    Parameters
    ----------
    net : torch.nn.Module
        The new model you will load the weight to. 
        The new model cannot contain batch_normal layer and the convolutional layer in it must have bias.
    state : ordered dict
        The dict contain weights in old model which contains batch_normal layers and convolutional layers with no bias.
    conv_names : list of str
        The name (or part name) list of your convolutional layers.
    bn_names : list of str
        The name (or part name) list of your batch_normal layers.

    Returns
    -------
    None. The new weights have been loaded to the new model.

    Notes
    -----
    None

    '''
    state = get_state(net, state_path, [], device)
    state = convert_prune_weight(state)
    name_param = net.named_parameters()
    
    weights = None
    gamma = None
    beta = None
    miu = None
    sigma = None
    for layer, layer_param in name_param:
        if is_contain_name(layer, conv_names, "weight"):
            weights = state[layer]
            layer_bn = get_bn_name(layer, name_map)
            if layer_bn.replace("weight", "weight") not in state:
                layer_param.data = weights
                print("ignore due to no bias", layer)
            else:         
                gamma = state[layer_bn.replace("weight", "weight")]
                beta = state[layer_bn.replace("weight", "bias")]
                miu = state[layer_bn.replace("weight", "running_mean")]
                sigma = torch.sqrt(state[layer_bn.replace("weight", "running_var")] + 1e-5)

                layer_param.data = weights*gamma.reshape(gamma.shape[0],1,1,1)/sigma.reshape(sigma.shape[0],1,1,1)
                sparsity = "{:.2f}%".format(100 - 100. * float(torch.sum(layer_param.data == 0))/ float(layer_param.data.nelement()))
                print("mix bn to:", layer, ", sparsity:", sparsity)

        elif is_contain_name(layer, conv_names, "bias"):
            layer_param.data = beta - miu*gamma/sigma
            print("add bias to:", layer)

        elif is_contain_name(layer, fc_names, "weight"):
            layer_param.data = state[layer]
            print("add wights to:", layer)

        elif is_contain_name(layer, fc_names, "bias"):
            layer_param.data = state[layer]
            print("add bias to:", layer)

        else:
            print("ignore", layer)

        # print(layer, layer_param.data.detach().cpu().numpy().ravel()[:5])

class Spatial_Postion:
    def __init__(self, input_shape, output_ch, kernel, padding, stride, group, need_group=""):
        left = padding
        right = kernel - padding
        input_ch, height, width = input_shape
        out_width = width + 2*padding - kernel + 1 
        out_height = height + 2*padding - kernel + 1 
        out_height = math.ceil( out_height * 1.0 / stride)
        out_width = math.ceil( out_width * 1.0 / stride)

        self.pos_pre = []
        self.pos_conv = []
        for i in range(out_height):
            pos_pre = []
            pos_conv = []
            for j in range(out_width):
                pos_pre.append([max(0, i*stride-left), min(i*stride+right, height), max(0, j*stride -left), min(j*stride+right, width)])
                pos_conv.append([max(0, left-i*stride), min(left+height-i*stride, kernel), max(0, left-j*stride), min(left+width-j*stride, kernel)])
            self.pos_pre.append(pos_pre)
            self.pos_conv.append(pos_conv)

        if need_group == "input":
            print("input neurons id coded in group")
            self.input_neuron = np.arange(input_ch * height * width).reshape(input_shape)
        else:
            tmp_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            self.input_neuron = np.arange(input_ch * height * width).reshape(tmp_input_shape).transpose((2, 0, 1))
        if need_group == "output":
            print("output neurons id coded in group")
            self.output_neuron = np.arange(output_ch * out_height * out_width).reshape((output_ch, out_height, out_width)) 
        else:
            tmp_output_shape = (out_height, out_width, output_ch)
            self.output_neuron = np.arange(output_ch * out_height * out_width).reshape(tmp_output_shape).transpose((2, 0, 1))

        self.group_in_ch = input_ch // group
        self.group_out_ch = output_ch // group

    def get_input_neuron(self, c, i, j):
        h1, h2, w1, w2 = self.pos_pre[i][j]
        ch1, ch2 = self.get_group_in_ch(c)
        return self.input_neuron[ch1:ch2, h1:h2, w1:w2].ravel()

    def get_conv(self, weights, c, i, j):
        h1, h2, w1, w2 = self.pos_conv[i][j]
        return weights[c, :, h1:h2, w1:w2].detach().numpy().ravel()

    def get_output_neuron(self, c, i, j):
        return self.output_neuron[c, i, j]

    def get_output_shape(self):
        return self.output_neuron.shape

    def get_group_in_ch(self, c):
        group = c // self.group_out_ch
        return group*self.group_in_ch, (group+1)*self.group_in_ch

    def info(self):
        print(self.pos_pre)
        print(self.pos_conv)

class Bias_Connect:
    def __init__(self):
        self.out_conns = 32
        self.conns = -1

    def get_id(self):
        self.conns += 1
        return self.conns // self.out_conns


def conv2d_connections(input_shape, module, num_chips=1, output_shape=[], prune=True, need_group=""):
    assert input_shape[0] // module.groups == module.weight.shape[1]
    print("prune", prune)
    padding = module.padding[0]
    stride = module.stride[0]
    kernel = module.kernel_size[0]
    group = module.groups // num_chips
    output_ch = module.weight.shape[0] // num_chips
    input_shape = (input_shape[0]//num_chips, *input_shape[1:])
    delay = 0

    spatial = Spatial_Postion(input_shape, output_ch, kernel, padding, stride, group, need_group)
    connections = []
    with_bias = module.bias != None
    for k in range(num_chips):
        conn = []
        weight = module.weight[output_ch*k: output_ch*(k+1)]
        if with_bias:
            conn_bias = []
            bias = module.bias[output_ch*k: output_ch*(k+1)].detach().numpy()
            bias_conn = Bias_Connect()
        channels, height, width = spatial.get_output_shape()
        for ch in range(channels):
            for h in range(height):
                for w in range(width):
                    # print(ch, h, w)
                    post_neuron_id = spatial.get_output_neuron(ch, h, w)
                    pre_neuron_ids = spatial.get_input_neuron(ch, h, w)
                    conv_weights = spatial.get_conv(weight, ch, h, w)
                    for n in range(len(pre_neuron_ids)):
                        if not prune or conv_weights[n] != 0:
                            conn.append((pre_neuron_ids[n], post_neuron_id, conv_weights[n], delay))
                    if with_bias and (not prune or bias[ch]!=0):
                        bias_neuron_id = bias_conn.get_id()
                        conn_bias.append((bias_neuron_id, post_neuron_id, bias[ch], delay))
                
        connections.append(conn)
        print("generate", len(conn), "conv_conn")
        if with_bias:
            connections.append(conn_bias)
            print("generate", len(conn_bias), "bias_conn")
    out_shape = spatial.get_output_shape()
    output_shape.append((out_shape[0]*num_chips, out_shape[1], out_shape[2]))
    return connections


def pool_connections(input_shape, module, num_chips=1, scale=100.0, output_shape=[], prune=True):
    stride = module.stride
    kernel = module.kernel_size
    assert stride == kernel
    input_shape = (input_shape[0]//num_chips, *input_shape[1:])
    output_ch = input_shape[0]
    weight = np.round(scale / (kernel * kernel))
    delay = 0

    conn = []
    if not prune or weight!=0:
        spatial = Spatial_Postion(input_shape, output_ch, kernel, 0, stride, group=output_ch)
        channels, height, width = spatial.get_output_shape()
        for ch in range(channels):
            for h in range(height):
                for w in range(width):
                    post_neuron_id = spatial.get_output_neuron(ch, h, w)
                    pre_neuron_ids = spatial.get_input_neuron(ch, h, w)
                    for n in range(len(pre_neuron_ids)):
                        conn.append((pre_neuron_ids[n], post_neuron_id, weight, delay))
    connections = []   
    for k in range(num_chips):
        connections.append(conn)
        print("generate", len(conn), "pool_conn")
    out_shape = spatial.get_output_shape()
    output_shape.append((out_shape[0]*num_chips, out_shape[1], out_shape[2]))
    return connections


def shortcut_connections(input_shape, num_chips=1, scale=1.0, prune=True):
    input_neurons = np.arange(np.array(input_shape).prod()//num_chips).tolist()
    weight = round(scale)
    delay = 0
    conn = np.empty((len(input_neurons), 4))
    if not prune or weight != 0:
        conn[:, 0] = input_neurons
        conn[:, 1] = input_neurons
        conn[:, 2] = weight
        conn[:, 3] = delay
    connections = [] 
    for k in range(num_chips):
        connections.append(conn)
        print("generate", len(conn), "shortcut_conn")
    return connections


def fc_connections(module, output_shape=[], prune=True):
    with_bias = module.bias != None
    weight = module.weight.detach().numpy()
    input_neurons = np.arange(weight.shape[1])
    output_neurons = np.arange(weight.shape[0])
    if with_bias:
        bias = module.bias.detach().numpy()
        bias_neurons = np.arange(weight.shape[0])    
        conn_bias = []
        bias_conn = Bias_Connect()
    delay = 0
    conn = []
    for o in output_neurons:
        for i in input_neurons:
            if not prune or weight[o][i]!=0:
                conn.append((i, o, weight[o][i], delay))
        if with_bias and (not prune or bias[o]!=0):
            bias_neuron_id = bias_conn.get_id()
            conn_bias.append((bias_neuron_id, o, bias[o], delay))
    print("generate", len(conn), "fc_conn")
    connections = []
    connections.append(conn)
    if with_bias:
        print("generate", len(conn_bias), "fc_bias")
        connections.append(conn_bias)
    output_shape.append((len(output_neurons),))
    return connections




# def convert_mobilenet():
#     model = moslicenetv10_nobn()
#     state_path = "checkpoint/2020-06-30T10:25moslicenetv10_prune_0.2/epoch_61_35.pth.tar"#"checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"#
#     name_map = [("blocks.0.0.conv_dw","blocks.0.0.bn1"), ("blocks.0.0.conv_pw","blocks.0.0.bn2"), ("conv_stem", "bn1"), ("conv_head", "bn2"),
#     ("conv_pwl","bn3"), ("conv_pw","bn1"), ("conv_dw", "bn2")]
#     mix_bn_to_conv(model, state_path, conv_names=["conv"], bn_names=["bn"], fc_names=["classifier"], name_map=name_map)
#     torch.save({'state_dict':model.state_dict()}, "checkpoint/0_pretrained_128/mixed_bn_mixed_prune.pth.tar")

#     # # convert_module(model.global_pool, "net.global_pool", 1, "net.conv_head", (1280, 4, 4))
#     # # convert_module(model.classifier, "net.classifier" , 1, "net.global_pool", (1280, 1, 1))
#     # convert_module(model, "net", 4, "input", (12, 64, 64))
#     # chips_merge("net.blocks.5.0.conv_dw_to_net.blocks.5.0.conv_pwl", 4, (192, 2, 2))
#     # chips_merge("net.blocks.5.0.conv_dw_to_net.blocks.5.0.conv_pwl_bias", 4, (192, 2, 2))

def save_connections(conns, name, path="connections/"):
    data = np.asarray(conns).astype(int)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, name), "wb") as f:
        pickle.dump(data, f)
    # np.savetxt(path+name+'.txt', data)
    pass

def chips_merge(name, num_chips, input_shape, path="connections/"):
    nums = np.array(input_shape).prod()
    for i in range(1, num_chips):
        full_path = path+name+"_chip"+str(i)
        with open(full_path, "rb") as f:
            data = pickle.load(f)
        data[:,1] += nums*i
        save_connections(data, name+"_chip"+str(i)+"_new")

if __name__ == "__main__":
    mute_prune_connections("connections", "connections_new")
    pass

#%%
    # convert_mobilenet()
    # # with open("connections/net.blocks.5.0.conv_pwl_to_net.blocks.5.2.conv_pw_chip0", "rb") as f:
    # #     data = pickle.load(f)
    # #     print(max(data[:, 0]))
    # pass

# %%
    # model = moslicenetv10_nobn()
    # state_path = "checkpoint/2020-06-28T02:26moslicenetv10_noblock/epoch_65_24.pth.tar"
    # print(model.state_dict().keys())
    # state = model.state_dict()
    # for name in state:
    #     print(name, state[name].shape)

    # mix_bn_to_conv(model, state_path, conv_names=["conv"], bn_names=["bn"], fc_names=["classifier"])
    # pass
    # spatial = Spatial_Postion((3, 4, 4), 3, 1, 0, 1, 1)
    # spatial.info()
    # channels, height, width = spatial.get_output_shape()  
    # bias = [1, 2, 3]
    # conn_bias = []
    # bias_conn = Bias_Connect()
    # for ch in range(channels):
    #     for h in range(height):
    #         for w in range(width):
    #             post_neuron_id = spatial.get_output_neuron(ch, h, w)
    #             pre_neuron_ids = spatial.get_input_neuron(ch, h, w)
    #             bias_neuron_id = bias_conn.get_id()
    #             conn_bias.append((bias_neuron_id, post_neuron_id, bias[ch], 0))
    #             print(post_neuron_id)
    #             print(pre_neuron_ids)
    # print(conn_bias)

    

    # %%
    # class Simple(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv = nn.Conv2d(9, 9, 3, stride = 1, padding = 1, groups=3)
    #         self.pool = nn.AvgPool2d(kernel_size=2)
    #         self.fc = nn.Linear(4, 10)
    #         self.weights_init()

    #     def forward(self, x):
    #         return self.conv(x)

    #     def weights_init(self):
    #         self.conv.weight = torch.nn.Parameter(torch.arange(1, np.array(self.conv.weight.shape).prod()+1).reshape(self.conv.weight.shape) * 0.1)
            
    #         self.conv.weight[:, :,:, ::2] = 0# self.conv.bias = torch.nn.Parameter(torch.zeros(4) * 0.1)

    #         self.conv.bias = torch.nn.Parameter(torch.arange(1, np.array(self.conv.bias.shape).prod()+1).reshape(self.conv.bias.shape) * 0.1)

    #         self.fc.weight = torch.nn.Parameter(torch.arange(1, np.array(self.fc.weight.shape).prod()+1).reshape(self.fc.weight.shape) * 0.1)
    
    # model = Simple()

    # # module = model.conv
    # # inputs = torch.ones((9, 4, 4))
    # # conns = conv2d_connections(inputs.shape, module, 3)

    # # module = model.pool
    # # inputs = torch.ones((3, 4, 4))
    # # conns = pool_connections(inputs.shape, module, 1)

    # # inputs2 = inputs.reshape(1, *inputs.shape)
    # # outputs2 = module(inputs2)
    # # print(outputs2)
    # # outputs = outputs2.reshape(outputs2.shape[1], outputs2.shape[2], outputs2.shape[3])
    
    # inputs = torch.ones((3, 4, 4))
    # conns = shortcut_connections(inputs.shape)
    # outputs = inputs

    # # inputs = torch.ones((4))
    # # module = model.fc
    # # conns = fc_connections(module)
    # # outputs = module(inputs)
    # # print(outputs)
    

    # from brian2 import *
    # eqs = '''
    # dv/dt = 0 / ms : 1
    # '''

    # dc = '''
    # dv/dt = 5 / ms : 1
    # '''

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

    # # S_bias = Synapses(bias_neurons, output_neurons, 'w : 1', on_pre='v_post += w', delay=0.0*ms)
    # # bias_conns = np.array(conns[1])
    # # S_bias.connect(i=bias_conns[:,0].astype(int).tolist(), j=bias_conns[:,1].astype(int).tolist())
    # # S_bias.w = bias_conns[:,2]

    # # store('initialized')
    # M_in = StateMonitor(input_neurons, 'v', record=True)
    # M_out = StateMonitor(output_neurons, 'v', record=True)
    # # M = StateMonitor(G, 'v', record=True)
    # run(2*ms)
    # print(M_out.v[:,-1].reshape(outputs.shape))
    

    # # plot(M_in.t/ms, M_in.v[0])
    # # xlabel('Time (ms)')
    # # ylabel('v')
    # # plot(M_out.t/ms, M_out.v[0])
    # # xlabel('Time (ms)')
    # # ylabel('v')

    
        
# %%














'''
class Spatial_Postion:
    def __init__(self, input_shape, output_ch, kernel, padding, stride, group):
        right = kernel // 2
        left = kernel - 1 - right
        input_ch, height, width = input_shape
        out_width = width + padding - right - left + padding 
        out_height = height + padding - right - left + padding 
        out_height = math.ceil( out_height * 1.0 / stride)
        out_width = math.ceil( out_width * 1.0 / stride)

        self.pos_pre = []
        self.pos_conv = []
        for i in range(out_height):
            pos_pre = []
            pos_conv = []
            for j in range(out_width):
                pos_pre.append([max(0, i*stride-left), min(i*stride+right+1, height), max(0, j*stride -left), min(j*stride+right+1, width)])
                pos_conv.append([max(0, left-i*stride), min(left+height-i*stride, kernel), max(0, left-j*stride), min(left+width-j*stride, kernel)])
            self.pos_pre.append(pos_pre)
            self.pos_conv.append(pos_conv)
        self.input_neuron = np.arange(input_ch * height * width).reshape(input_shape)
        self.output_neuron = np.arange(output_ch * out_height * out_width).reshape((output_ch, out_height, out_width)) 

        self.group_in_ch = input_ch // group
        self.group_out_ch = output_ch // group

    def get_input_neuron(self, c, i, j):
        h1, h2, w1, w2 = self.pos_pre[i][j]
        ch1, ch2 = self.get_group_in_ch(c)
        return self.input_neuron[ch1:ch2, h1:h2, w1:w2].ravel()

    def get_conv(self, weights, c, i, j):
        h1, h2, w1, w2 = self.pos_conv[i][j]
        return weights[c, :, h1:h2, w1:w2].detach().numpy().ravel()

    def get_output_neuron(self, c, i, j):
        return self.output_neuron[c, i, j]

    def get_output_shape(self):
        return self.output_neuron.shape

    def get_group_in_ch(self, c):
        group = c // self.group_out_ch
        return group*self.group_in_ch, (group+1)*self.group_in_ch

    def info(self):
        print(self.pos_pre)
        print(self.pos_conv)

class Bias_Connect:
    def __init__(self):
        self.out_conns = 32
        self.conns = -1

    def get_id(self):
        self.conns += 1
        return self.conns // self.out_conns


def conv2d_connections(input_shape, module, num_chips):
    assert input_shape[0] == module.weight.shape[1]
    padding = module.padding[0]
    stride = module.stride[0]
    group = module.groups // num_chips
    kernel = module.kernel_size[0]
    output_ch = module.weight.shape[0] // num_chips
    delay = 0

    spatial = Spatial_Postion(input_shape, output_ch, kernel, padding, stride, group)
    connections = []
    for k in range(num_chips):
        conn = []
        conn_bias = []
        weight = module.weight[output_ch*k: output_ch*(k+1)]
        bias = module.bias[output_ch*k: output_ch*(k+1)]
        bias_conn = Bias_Connect()
        channels, height, width = spatial.get_output_shape()
        for ch in range(channels):
            for h in range(height):
                for w in range(width):
                    # print(ch, h, w)
                    post_neuron_id = spatial.get_output_neuron(ch, h, w)
                    pre_neuron_ids = spatial.get_input_neuron(ch, h, w)
                    conv_weights = spatial.get_conv(weight, ch, h, w)
                    for n in range(len(pre_neuron_ids)):
                        if conv_weights[n] != 0:
                            conn.append((pre_neuron_ids[n], post_neuron_id, conv_weights[n], delay))
                    bias_neuron_id = bias_conn.get_id()
                    conn_bias.append((bias_neuron_id, post_neuron_id, bias[ch].item(), delay))
                
        connections.append(conn)
        connections.append(conn_bias)
        print("generate", len(conn), "conv_conn and", len(conn_bias), "bias_conn")
    return connections



def mix_bn_to_conv(net, state_path, conv_names=["conv"], bn_names=["bn"], fc_names=["fc"]):
    # 
    # Combine the batch_normal layer with the convolutional layer.

    # Parameters
    # ----------
    # net : torch.nn.Module
    #     The new model you will load the weight to. 
    #     The new model cannot contain batch_normal layer and the convolutional layer in it must have bias.
    # state : ordered dict
    #     The dict contain weights in old model which contains batch_normal layers and convolutional layers with no bias.
    # conv_names : list of str
    #     The name (or part name) list of your convolutional layers.
    # bn_names : list of str
    #     The name (or part name) list of your batch_normal layers.

    # Returns
    # -------
    # None. The new weights have been loaded to the new model.

    # Notes
    # -----
    # None

    # 
    state = get_state(net, state_path, [])
    state = convert_prune_weight(state)
    state = list(state.items())
    name_param = net.named_parameters()
    
    ind_state = 0
    weights = None
    gamma = None
    beta = None
    miu = None
    sigma = None
    for layer, layer_param in name_param:
        if ind_state >= len(state):
            break
        print(layer)
        if is_contain_name(layer, conv_names, "weight"):
            while layer != state[ind_state][0]:
                print("drop", state[ind_state][0])
                ind_state += 1
            weights = state[ind_state][1]
            ind_state += 1
            if is_contain_name(state[ind_state][0], bn_names, "weight"):
                gamma = state[ind_state][1]
                ind_state += 1
            else:
                error_report("cannot find bn weight")
            if is_contain_name(state[ind_state][0], bn_names, "bias"):
                beta = state[ind_state][1]
                ind_state += 1
            else:
                error_report("connot find bn bias")
            if is_contain_name(state[ind_state][0], bn_names, "running_mean"):
                miu = state[ind_state][1]
                ind_state += 1
            else:
                error_report("connot find bn mean")
            if is_contain_name(state[ind_state][0], bn_names, "running_var"):
                sigma = torch.sqrt(state[ind_state][1] + 1e-5)
                ind_state += 1
            else:
                error_report("connot find bn var")
            layer_param.data = weights*gamma.reshape(gamma.shape[0],1,1,1)/sigma.reshape(sigma.shape[0],1,1,1)
            print("mix bn to:", layer)
        elif is_contain_name(layer, conv_names, "bias"):
            layer_param.data = beta - miu*gamma/sigma
            print("add bias to:", layer)
        elif is_contain_name(layer, fc_names, "weight"):
            while layer != state[ind_state][0]:
                print("drop", state[ind_state][0])
                ind_state += 1
            layer_param.data = state[ind_state][1]
            ind_state += 1
            print("add wights to:", layer)
        elif is_contain_name(layer, fc_names, "bias"):
            while layer != state[ind_state][0]:
                print("drop", state[ind_state][0])
                ind_state += 1
            layer_param.data = state[ind_state][1]
            ind_state += 1
            print("add bias to:", layer)
        else:
            ind_state += 1
            print("ignore", layer)
'''
