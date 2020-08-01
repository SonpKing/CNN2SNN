"""
(32*1024-4)/(4+2*n+conn*2)
"""
import pickle as pk
import math
import numpy as np

def count_nodes(layers, avg_conn, Neu):
    assert len(layers) == len(avg_conn) #断言函数，否则终止程序
    length = len(layers)
    # print(length)
    nodes = []
    neus = []
    nodes.append(math.ceil(layers[length - 1] / Neu)) #对浮点数向上取整，加到nodes末尾 Neu=256
    neus.append(Neu)
    for i in range(length - 2, -1, -1):  # 7 - 0
        neurons = layers[i]
        conn = avg_conn[i]
        # print(nodes[-1],neus[-1])
        neu = (30 * 1024 - 4) // (4 + 2 * nodes[-1] + conn * 2)  # //除数取整数,nodes[-1]为该数组倒数第一个值
        # print(neurons,conn,neu)
        neu = min(neu, Neu)
        # print(neu)
        # print(neurons / neu)
        nodes.append(math.ceil(neurons / neu))
        neus.append(neu)
    for j in neus:
        assert j <= 256
    return nodes[::-1], neus[::-1] #翻转

def count_neucon(filelist):
    layers = []
    avg_con = []
    assert len(filelist) != 0

    f = open(filelist[0], 'rb') #one-to-one
    cons = pk.load(f)  #从文件变为python对
    dst = set() #创建一个无序不重复元素集
    for con in cons:
        dst.add(con[1])
    layers.append(len(dst))
    f.close()
    for file in filelist[1:]:
        f = open(file, 'rb')
        cons = pk.load(f)
        print(len(cons))
        dst = set()
        for con in cons:
            dst.add(con[1])
        avg_con.append(int(math.ceil(len(cons) / float(layers[-1]))))
        layers.append(len(dst))

        f.close()
    avg_con.append(1)
    return layers, avg_con


def count_neucon2(filelist):
    layers = []
    max_con = []
    assert len(filelist) != 0
    f = open(filelist[0], 'rb')
    cons = pk.load(f, encoding='iso-8859-1')
    # cons=pk.load(f)

    dst = set()
    for con in cons:
        dst.add(con[1])
    layers.append(len(dst) + 1)
    # print(layers)
    f.close()

    for file in filelist[1:]:
        f = open(file, 'rb')
        cons = pk.load(f, encoding='iso-8859-1')
        print(len(cons))
        dst = set()
        maxsou = -1
        for con in cons:
            dst.add(con[1])
            if con[0] > maxsou:
                maxsou = int(con[0])

        layers.append(len(dst) + 1)
        cnt_con = np.zeros(((maxsou + 1),))
        for con in cons:
            cnt_con[int(con[0])] += 1
        maxnum = np.max(cnt_con)
        max_con.append(int(maxnum))
        f.close()
    max_con.append(1)
    return layers, max_con

def neuron_link(connfiles, netDepth, layers, nodes):
    neuron_id = []  #神经元的node号码
    neuron_dst = []  #神经元的目的神经元
    node_link = []
    # link_to = []
    for i in range(netDepth - 2):
        x_loop = []
        for j in range(nodes[i]):
            x_loop.append([])
        node_link.append(x_loop)

    # for i in range(netDepth - 2):
    #     x_loop = []
    #     for j in range(nodes[i + 1]):
    #         x_loop.append([])
    #     link_to.append(x_loop)
    # link_to.append([[]])
    # print(link_to)
    for i in range(netDepth-1):
        neuron_num = int(math.ceil(layers[i] / float(nodes[i])))
        # print('neuron_num:',neuron_num)
        neuron_ii = []
        layer_dst = []
        for j in range(layers[i]):
            neuron_ii.append(j//neuron_num)
            layer_dst.append([])
        neuron_id.append(neuron_ii)
        neuron_dst.append(layer_dst)
        # print(neuron_id)

        if i != netDepth-2:
            f = open(connfiles[i+1],'rb')
            # print(f)
            # conn = np.array(pk.load(f))
            conn = np.array(pk.load(f,encoding='iso-8859-1'))
            for line in conn:
                src = int(line[0])
                dst = int(line[1])
                neuron_dst[i][src].append(dst)

    for i in range(netDepth-2):
        # neuron_num = int(math.ceil(layers[i+1] / float(nodes[i+1]))) # 下一层
        for j in range(layers[i]):
            src_id = neuron_id[i][j]
            for k in range(len(neuron_dst[i][j])):
                dst = neuron_dst[i][j][k]
                dst_id = neuron_id[i+1][dst]

                if dst_id not in node_link[i][src_id]:
                    node_link[i][src_id].append(dst_id)
                # if src_id not in link_to[i][dst_id]:
                #     link_to[i][dst_id].append(src_id)

    # for i in range(nodes[-1]):
    #     node_link[netDepth - 2][i].append(0)
        # link_to[netDepth - 2][0].append(i)
    return node_link

if __name__ == "__main__":
    # connfiles = [
    #     "conn2/input_to_fc1",
    #     "conn2/fc1_to_fc2",
    #     'conn2/fc2_to_out'
    # ]
    # connfiles = [
    #     "connections/0_to_2",
    #     "connections/2_to_4_test",
    #     'connections/4_to_6',
    #     'connections/6_to_8',
    #     'connections/8_to_10',
    #     'connections/10_to_12'
    # ]
    connfiles = [
        "connections2/0_to_2",
        "connections2/2_to_4",
        'connections2/4_to_6',
        'connections2/6_to_8',
        'connections2/8_to_10',
        'connections2/10_to_12',
        'connections2/12_to_14',
        'connections2/14_to_16',
        'connections2/16_to_18'
    ]
    layers, avg_conn = count_neucon2(connfiles)  #####
    print("layers:", layers)
    print("avg_conn:", avg_conn)

    nodes, neus = count_nodes(layers, avg_conn, 256)  # only nodes will be used
    print('nodes:', nodes)
    print('neus:', neus)

    node_link = neuron_link(connfiles, 10, layers, nodes)
    print(node_link)
    # print(link_to)