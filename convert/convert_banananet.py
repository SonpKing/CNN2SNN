import math
import pickle
import os
from models.BananaNet import Pool_Scale, Blocks2
from models import Scale
from .convert import *

def convert_module(module, pre_name, num_chips, last_layer, input_shape, prune, need_group=""):
    '''
    return last_layer_name, last_output_shape, num_chips
    '''
    name = pre_name
    print("start convert", name)
    # print(input_shape)
    if isinstance(module, nn.Conv2d):
        output_shape = []
        conns = conv2d_connections(input_shape, module, num_chips, output_shape, prune=prune, need_group=need_group)
        with_bias = module.bias!=None
        i = 0
        while i < len(conns):
            save_connections(conns[i], last_layer+"_to_"+name+"_chip"+str(i//2))
            i += 1
            if with_bias:
                save_connections(conns[i+1], last_layer+"_to_"+name+"_bias_chip"+str(i//2))
                i += 1
        last_layer, input_shape = name, output_shape[0]

    elif isinstance(module, Pool_Scale):
        output_shape = []
        pool_scale = module.scale.weight.detach().numpy()
        print("pool_scale", pool_scale)
        conns = pool_connections(input_shape, module.pool, num_chips, scale = pool_scale, output_shape=output_shape, prune=prune, need_group=need_group)
        for i in range(len(conns)):
            save_connections(conns[i], last_layer+"_to_"+name+"_chip"+str(i))
        last_layer, input_shape = name, output_shape[0]


    elif isinstance(module, nn.Linear):
        with_bias = module.bias != None
        output_shape = []
        conns = fc_connections(module, output_shape, prune=prune)
        save_connections(conns[0], last_layer+"_to_"+name+"_chip"+str(0))
        if with_bias:
            save_connections(conns[1], last_layer+"_to_"+name+"_bias_chip"+str(0))
        last_layer, input_shape = name, output_shape[0]


    elif isinstance(module, Blocks2):
        last_need_group = need_group == "input"
        for child_name, child_mod in module.named_children():
            child_name = name + "." + child_name
            if isinstance(child_mod, Pool_Scale):
                last_layer, input_shape, num_chips = convert_module(child_mod, child_name, num_chips, last_layer, input_shape, prune=prune, need_group="output")
                last_need_group = True
            elif isinstance(child_mod, nn.Conv2d) and last_need_group:
                last_layer, input_shape, num_chips = convert_module(child_mod, child_name, num_chips, last_layer, input_shape, prune=prune, need_group="input")
                last_need_group = False
            else:
                last_layer, input_shape, num_chips = convert_module(child_mod, child_name, num_chips, last_layer, input_shape, prune=prune)


    elif isinstance(module, nn.Sequential) or name=="net":
        last_need_group = False
        for name, mod in module.named_children():
            name = pre_name + "."+ name
            if last_need_group:
                last_layer, input_shape, num_chips = convert_module(mod, name, num_chips, last_layer, input_shape, prune=prune, need_group="input")
                last_need_group = False
            else:
                last_layer, input_shape, num_chips = convert_module(mod, name, num_chips, last_layer, input_shape, prune=prune)
            if isinstance(mod, Blocks2):
                last_need_group = True
        
    else:
        print("ignore", name)

    print(input_shape)

    return last_layer, input_shape, num_chips



def normalise_module(module, pre_name, max_acts, last_act, compensation=1.0):
    '''
    return last_layer_name, last_output_shape, num_chips
    '''
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, Scale):
        print("normalise", pre_name, last_act, max_acts[pre_name])
        print(last_act / max_acts[pre_name] * compensation)
        module.weight.data *= last_act / max_acts[pre_name] * compensation
        if module.bias != None:
            module.bias.data *= 1.0 / max_acts[pre_name]
        last_act = max_acts[pre_name]
        

    elif isinstance(module, Blocks2):#isinstance(module, DepthwiseConv) or isinstance(module, InvertedResidual):
        old_last_act = last_act
        if pre_name:
            pre_name += '.'
        for child_name, child_mod in module.named_children():
            child_name = pre_name + child_name
            last_act = normalise_module(child_mod, child_name, max_acts, last_act, compensation)

    elif isinstance(module, nn.Sequential) or not pre_name:
        if pre_name:
            pre_name += '.'
        for child_name, child_mod in module.named_children(): 
            child_name = pre_name + child_name
            last_act = normalise_module(child_mod, child_name, max_acts, last_act, compensation)
        
    else:
        print("ignore", pre_name)

    return last_act
