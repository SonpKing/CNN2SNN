# -*- coding: utf-8 -*-
class node_input(object):
    #address unit is word = 4 Bytes
    def __init__(self, node_number, cons, neurons={}, grid_size=24):
        import math
        self.grid_size = grid_size
        self.body_pack_head = ((self.grid_size - 1) << 38) | (0b1 << 32)
        self.choosebits = math.ceil(math.log(grid_size, 2))
        self.set_nodenum(node_number)
        self.init(neurons, cons)

    def set_nodenum(self, node_number):
        self.x = node_number // self.grid_size
        self.y = node_number % self.grid_size

    def get_headpack(self): #from (grid_size,grid_size) to (x,y)       dst_port: 0 -> 1,  2 -> 2, 3 -> 4
        tmp=(self.x<<18)|(self.y<<12)|(self.grid_size<<6)|(self.grid_size-1)
        # if self.x==self.grid_size-1 and self.y==self.grid_size-1:
        #     return (self.y<<38)|(0b10<<36)|(0b1<<32)|(0b1<<29)|(0x1<<24)|tmp
        # elif self.x==self.grid_size-1:
        #     return (self.y<<38)|(0b10<<36)|(0b1<<32)|(0b1<<29)|(0x2<<24)|tmp
        # else:
        #     return (self.y<<38)|(0b10<<36)|(0b1<<32)|(0b1<<29)|(0x4<<24)|tmp
        if self.x==self.grid_size-1 and self.y==self.grid_size-1:
            return ((self.grid_size-1)<<38)|(0b10<<36)|(0b1<<32)|(0b1<<29)|(0x1<<24)|tmp
        elif self.x==self.grid_size-1:
            return ((self.grid_size-1)<<38)|(0b10<<36)|(0b1<<32)|(0b1<<29)|(0x2<<24)|tmp
        else:
            return ((self.grid_size-1)<<38)|(0b10<<36)|(0b1<<32)|(0b1<<29)|(0x4<<24)|tmp
    def get_bodypackhead(self):
        # return (self.y<<38)|(0b1<<32)
        return self.body_pack_head
    def get_tailpackhead(self):
        # return (self.y<<38)|(0b01<<36)|(0b1<<32)
        return ((self.grid_size-1)<<38)|(0b01<<36)|(0b1<<32)

    def set_neurons(self, neurons={}):
        self.neurons = neurons
        self.neu_r = {}
        self.dst = set()
        for n in self.neurons:
            self.neu_r[self.neurons[n]] = n
            self.dst.add(self.neurons[n])

    def set_src(self, cons):  #original connection
        self.src = set()
        for con in cons:
            if con[1] in self.dst:
                self.src.add(con[0])

    def init(self, neurons={}, cons=[]):
        self.set_neurons(neurons)
        self.set_src(cons)

    def gen_input(self, src_id_set, connections):
        newsrc = src_id_set & self.src
        res = []
        for sr in newsrc:
            for ds in self.dst:
                tmp = (int(sr) << 32) | int(ds)
                # tmp = (sr << 32) | ds
                if tmp in connections:
                    wgt = connections[tmp]
                    n_id = self.neu_r[ds]
                    res.append((n_id << 8) | wgt)
        return res


def spike_time(spikes, maxtime):
    times = {}
    for i in range(maxtime + 1)[1:]:
        times[i] = set()
    for spike in spikes:
        neu_id = spike[0]
        ts = spike[1]
        for t in ts:
            if t in times:
                times[t].add(neu_id)
            else:
                times[t] = set()
                times[t].add(neu_id)
    return times


def change_format(cons):
    res = {}
    for con in cons:
        tmp = (int(con[0]) << 32) | int(con[1])
        # TODO:要乘以1000
        res[tmp] = int(con[2]) if con[2] >= 0 else 2**8 + int(con[2])
    return res


def gen_inputdata(cons,
                  spikes,
                  input_node_map,
                  maxtime,
                  filename="input.txt",
                  row_file="row.txt",
                  grid_size=24):
    '''
    cons:连接文件
    spikes:输入脉冲文件
    input_node_map:第一层神经元的分布情况
    maxtime：运行时间
    return：input.txt, row.txt
    '''
    f = open(filename, "w")
    inputnodes = []
    for k in input_node_map:
        inputnodes.append(
            node_input(k, cons, input_node_map[k], grid_size=grid_size))
    connections = change_format(cons)
    times = spike_time(spikes, maxtime)
    count = 0
    res = []
    for i in range(maxtime + 1)[1:]:
        res.append(count)
        neu_set = times[i]
        for node in inputnodes:
            tmp = node.gen_input(neu_set, connections)
            tmp2 = node.get_headpack()
            body_pack_head = node.get_bodypackhead()
            tail_pack_head = node.get_tailpackhead()
            ss = "%011x" % tmp2  #head
            f.write(ss + '\n')
            count += 1
            flag = 0
            sig = 0
            if len(tmp) != 0:
                for wt in tmp[:len(tmp) - 1]:
                    if flag == 0:
                        flag = 1
                        sig = wt
                    else:
                        flag = 0
                        ss = "%011x" % (((sig << 16) | wt) + body_pack_head)
                        f.write(ss + '\n')
                        count += 1
                if flag == 1:
                    ss = "%011x" % (((sig << 16) | tmp[-1]) + tail_pack_head)
                    f.write(ss + '\n')
                    count += 1
                else:
                    sig = tmp[-1]
                    ss = "%011x" % (((sig << 16) | 0) + tail_pack_head)
                    f.write(ss + '\n')
                    count += 1
            else:
                ss = "%011x" % tail_pack_head
                f.write(ss + '\n')
                count += 1
    res.append(count)
    f.close()
    frow = open(row_file, "w")
    ttt = 1
    for i in res[1:]:
        frow.write(str(hex(i))[2:] + '\n')  #rownum  time
        # frow.write(str(hex(i))[2:]+'  '+str(ttt)+'\n')    #rownum  time
        ttt += 1
    frow.close()

def gen_inputdata_list(cons,
                  spikes,
                  input_node_map,
                  maxtime,
                  grid_size=24):
    '''
    cons:连接文件
    spikes:输入脉冲文件
    input_node_map:第一层神经元的分布情况
    maxtime：运行时间
    return：input.txt, row.txt
    '''
    input_list = []
    row_list = []
    inputnodes = []
    for k in input_node_map:
        inputnodes.append(
            node_input(k, cons, input_node_map[k], grid_size=grid_size))
    connections = change_format(cons)
    times = spike_time(spikes, maxtime)
    count = 0
    res = []
    for i in range(maxtime + 1)[1:]:
        res.append(count)
        neu_set = times[i]
        for node in inputnodes:
            tmp = node.gen_input(neu_set, connections)
            tmp2 = node.get_headpack()
            body_pack_head = node.get_bodypackhead()
            tail_pack_head = node.get_tailpackhead()
            ss = "%011x" % tmp2  #head
            input_list.append(ss)
            
            count += 1
            flag = 0
            sig = 0
            if len(tmp) != 0:
                for wt in tmp[:len(tmp) - 1]:
                    if flag == 0:
                        flag = 1
                        sig = wt
                    else:
                        flag = 0
                        ss = "%011x" % (((sig << 16) | wt) + body_pack_head)
                        input_list.append(ss)
                        count += 1
                if flag == 1:
                    ss = "%011x" % (((sig << 16) | tmp[-1]) + tail_pack_head)
                    input_list.append(ss)
                    count += 1
                else:
                    sig = tmp[-1]
                    ss = "%011x" % (((sig << 16) | 0) + tail_pack_head)
                    input_list.append(ss)
                    count += 1
            else:
                ss = "%011x" % tail_pack_head
                input_list.append(ss)
                count += 1
    res.append(count)
    
    
    ttt = 1
    for i in res[1:]:
        row_list.append(str(hex(i))[2:])
        # frow.write(str(hex(i))[2:]+'  '+str(ttt)+'\n')    #rownum  time
        ttt += 1
    
    return input_list, row_list
    


if __name__ == '__main__':
    import pickle
    import numpy as np
    # 加载输入层
    fpkl = open('./connection/input_conv1.pkl', 'rb')
    in_conv1 = pickle.load(fpkl, encoding='iso-8859-1')
    fpkl.close()
    # 加载spikes
    fspk = open('./input/spikeTrains.pkl', 'rb')
    spikes = pickle.load(fspk, encoding='iso-8859-1')
    fspk.close

    input_node_map = {}

    # node分配列表
    layer = []
    node_list = []
    for x in range(8):
        for y in range(16, 24):
            layer.append([x, y])
    node_list.append(layer)
    # 跟据node数量，确定每个node分配的神经元数量
    layerWidth = 6272  #第一层神经元总个数
    neuron_num = layerWidth // len(node_list[0])

    for line in in_conv1:
        src = int(line[0])
        dst = int(line[1])
        node_x = node_list[0][dst // neuron_num][0]  # node_list[0], 第一层
        node_y = node_list[0][dst // neuron_num][1]
        nodenumber = node_x * 24 + node_y
        if not nodenumber in input_node_map.keys():
            input_node_map[nodenumber] = {}
        input_node_map[nodenumber].update({dst % neuron_num: dst})

    # print(input_node_map[0])
    # print(input_node_map[1])
    # 生成input
    change_format(in_conv1)
    gen_inputdata(in_conv1, spikes, input_node_map, 5, './input/input.txt',
                  './input/row.txt')
    print('done')
