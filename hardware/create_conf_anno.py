import pickle
import math
import time
from mapping import node_alloc as alloc
from mapping import gen_input as gen_in
from mapping import count
from mapping import map2
import os
import numpy as np


def get_headpack(x, y):
    x1 = x % 24
    y1 = y % 24
    tmp = (x << 18) | (y << 12) | (48 << 6) | (47) ###(24,23)input
    if x1 == 23:
        return (y1 << 38) | (0b10 << 36) | (0b1 << 32) | (0b1 << 29) | (0x1 << 24) | tmp
    else:
        return (y1 << 38) | (0b10 << 36) | (0b1 << 32) | (0b1 << 29) | (0x4 << 24) | tmp


def get_bodypackhead(y):
    y1 = y % 24
    return (y1 << 38) | (0b1 << 32)


def get_tailpackhead(y):
    y1 = y % 24
    return (y1 << 38) | (0b01 << 36) | (0b1 << 32)

def create_config():
    neurontype = 1
    grid_size = 64
    leak = 0
    vth = [100] + [60]*21
    reset_mode = [1] + [1]*21
    delay = [0xaaaaaaaa, 0xaaaaaaaa]

    
    netDepth = len(vth) + 1
    leaksign = -1
    board_num = 1  # 板子个数
    childboard_num = 3  # 单块板子的子板个数
    # chip_num = [1, 1, 1] #单块子板一条边的芯片个数，单块子板有chip_num*chip_num块芯片
    node_num = 24  # 单块芯片一条边的节点个数，单块芯片有node_num* node_num个节点
    neuron_num = 256  # 每个节点神经元数目

    connfiles = [
        'connections_anno_60_7/start_to_input_chip0',  # 1 3072
        'connections_anno_60_7/input_to_net.conv_stem_chip0',  # 3072 3840
        'connections_anno_60_7/net.conv_stem_to_net.blocks.0.0.conv_dw_chip0',  # 3840 3840
        'connections_anno_60_7/net.blocks.0.0.conv_dw_to_net.blocks.0.0.conv_pw_chip0',  # 3840 6144
        'connections_anno_60_7/net.blocks.0.0.conv_pw_to_net.blocks.1.0.conv_pw_chip0',  # 6144 12023
        'connections_anno_60_7/net.blocks.1.0.conv_pw_to_net.blocks.1.0.conv_dw_chip0',  # 12023 3008
        'connections_anno_60_7/net.blocks.1.0.conv_dw_to_net.blocks.1.0.conv_pwl_chip0',  # 3008 2048
        'connections_anno_60_7/net.blocks.1.0.conv_pwl_to_net.blocks.1.1.conv_pw_chip0',  # 2048 7488
        'connections_anno_60_7/net.blocks.1.1.conv_pw_to_net.blocks.1.1.conv_dw_chip0',  # 7488 7488
        'connections_anno_60_7/net.blocks.1.1.conv_dw_to_net.blocks.1.1.conv_pwl_chip0',  # 7488 2048
        'connections_anno_60_7/net.blocks.1.1.conv_pwl_to_net.blocks.2.0.conv_pw_chip0',  # 2048 6012
        'connections_anno_60_7/net.blocks.2.0.conv_pw_to_net.blocks.2.0.conv_dw_chip0',  # 6012 1504
        'connections_anno_60_7/net.blocks.2.0.conv_dw_to_net.blocks.2.0.conv_pwl_chip0',  # 1504 640
        'connections_anno_60_7/net.blocks.2.0.conv_pwl_to_net.blocks.2.1.conv_pw_chip0',  # 640 3120
        'connections_anno_60_7/net.blocks.2.1.conv_pw_to_net.blocks.2.1.conv_dw_chip0',  # 3120 3120
        'connections_anno_60_7/net.blocks.2.1.conv_dw_to_net.blocks.2.1.conv_pwl_chip0',  # 3120 640
        'connections_anno_60_7/net.blocks.2.1.conv_pwl_to_net.blocks.2.2.conv_pw_chip0',  # 640 1888
        'connections_anno_60_7/net.blocks.2.2.conv_pw_to_net.blocks.2.2.conv_dw_chip0',  # 1888 1888
        'connections_anno_60_7/net.blocks.2.2.conv_dw_to_net.blocks.2.2.conv_pwl_chip0',  # 1888 640
        'connections_anno_60_7/net.blocks.2.2.conv_pwl_to_net.conv_head_chip0',  # 640 4064
        'connections_anno_60_7/net.conv_head_to_net.global_pool_chip0',  # 4064 254
        'connections_anno_60_7/net.global_pool_to_net.classifier_chip0'  # 254 8
    ]
    tmp = []
    for connfile in connfiles:
        with open(connfile, "rb") as f:
            data = pickle.load(f)
        tmp.append(np.max(data[:, 0]) + 2)
    with open(connfiles[-1], "rb") as f:
        data = pickle.load(f)
    tmp.append(np.max(data[:, 1]) + 2)
    tmp.append(1)
    layerWidth = tmp
    print(layerWidth)
    
    print("start")
    flag = 0
    firstlayer = []

    # mapping
    layers, avg_conn = count.count_neucon2(connfiles)  #####
    print("layers:", layers)
    print("avg_conn:", avg_conn)

    nodes, neus = count.count_nodes(layers, avg_conn, neuron_num)  # only nodes will be used
    print('nodes:', nodes)
    print('neus:', neus)

    node_link = count.neuron_link(connfiles, netDepth, layers, nodes)
    print(node_link)
    layer_num = len(layers)
    layer_id = 0
    print("layer_num:", layer_num)

    board_id = 1
    forward_node = []
    while board_id <= board_num:
        childboard_id = 1

        while childboard_id <= childboard_num:
            id1 = layer_id + 1
            ID = '%d' % board_id + '_' + '%d' % childboard_id
            print(ID)
            print(childboard_id)
            # print(chip_num[childboard_id-1])
            # nodelist, zero_ref, layer_id = mapping.mapping_board(nodes, neus, layer_num, layer_id, chip_num[childboard_id-1])
            nodelist, zerolist, layer_id, forward_res = map2.map_chip11(nodes, node_link, layer_num, layer_id,
                                                                       forward_node)
            forward_node = forward_res
            print(forward_node)
            if id1 == 1:
                firstlayer = nodelist[0]
                # f = open('conn2/input_to_fc1', 'rb')
                # get_input(f, layerWidth, firstlayer)

            print("mapping done")
            print(zerolist)
            cnt_node = 0
            for i in range(len(nodelist)):
                cnt_node += len(nodelist[i])

            print("cnt_node:", cnt_node)

            ########################已经全部映射完成
            if (layer_id >= layer_num):
                flag = 1
                linkout = []  #######################(最后输出节点与最后一层的链接文件)
                for i in range(layers[-1]):
                    linkout.append((i, 0, i, 0))
                fw = open('config/linkout', 'wb')
                pickle.dump(linkout, fw)  # 保存到文件中
                fw.close()
                connfiles.append('config/linkout')

                ## output
                output = [[49, 49]]
                nodelist.append(output)
            else:
                nodelist.append(forward_node)

            id2 = layer_id + 1
            print("id2：", id2)
            # print(layerWidth[1:])
            # print(layerWidth[id1:id2 + 1])
            # print(len(connfiles))
            # print(connfiles[1:])
            # print(len(layerWidth))
            # print(connfiles[id1:id2])

            alloc.buildNetwork(ID, connfiles[id1:id2], id2 - id1 + 1, layerWidth[id1:id2 + 1], nodelist,
                               zerolist, delay, vth[id1 - 1:id2 - 1], leak, reset_mode[id1 - 1:id2 - 1], leaksign)
            print("config done")
            
            str1 = 'config/pickle/connfiles' + ID
            print(str1)
            fw = open(str1, 'wb')
            pickle.dump(connfiles, fw)
            fw.close()

            str2 = 'config/pickle/layerWidth' + ID
            fw = open(str2, 'wb')
            pickle.dump(layerWidth, fw)
            fw.close()

            str3 = 'config/pickle/nodelist' + ID
            fw = open(str3, 'wb')
            pickle.dump(nodelist, fw)
            fw.close()
            
            c_head = '40000'
            str4 = os.path.join("config", ID + "clear.txt")
            f = open(str4, "w")
            for i in range(id2 - id1):
                for x, y in nodelist[i]:
                    tmp = get_headpack(x, y)
                    body_pack_head = get_bodypackhead(y)
                    tail_pack_head = get_tailpackhead(y)
                    ss = "%011x" % tmp  # head
                    f.write(c_head + ss + '\n')
                    tmp = (0b1 << 31) | 0x0
                    ss = "%011x" % (tmp + body_pack_head)
                    f.write(c_head + ss + '\n')
                    tmp = 2
                    ss = "%011x" % (tmp + tail_pack_head)
                    f.write(c_head + ss + '\n')
            f.close()

            str5 = os.path.join("config", ID + "enable.txt")
            f = open(str5, "w")
            for i in range(id2 - id1):
                # print("i:", i)
                for x, y in nodelist[i]:
                    tmp = get_headpack(x, y)
                    body_pack_head = get_bodypackhead(y)
                    tail_pack_head = get_tailpackhead(y)
                    ss = "%011x" % tmp  # head
                    f.write(c_head + ss + '\n')
                    tmp = (0b1 << 31) | 0x0
                    ss = "%011x" % (tmp + body_pack_head)
                    f.write(c_head + ss + '\n')
                    tmp = 1
                    ss = "%011x" % (tmp + tail_pack_head)
                    f.write(c_head + ss + '\n')
            f.close()

            for node in nodelist:
                print(node)
            # print(nodelist)

            if (flag == 1):
                break
            childboard_id += 1
        # print(childboard_id)
        ########################已经全部映射完成
        if (flag == 1):
            break

        board_id += 1

    # ------------------------------------- input ----------------------------------------------总共只要一个
    # 加载输入层
    f = open('connections/start_to_input_chip0', 'rb')
    in_conv1 = pickle.load(f)
    f.close()

    # 加载spikes
    t1 = []
    for i in range(3073):  #### 输入层neuron number
        t1.append([i, [1]])

    print("in_conv1 len: ", len(in_conv1))
    times = time.time()

    input_node_map = {}
    neuron_num = int(math.ceil(layerWidth[1] / float(len(firstlayer))))

    interval = 1  #

    for line in in_conv1:
        src = int(line[0])
        dst = int(line[1])
        node_x = firstlayer[dst // neuron_num][0]
        node_y = firstlayer[dst // neuron_num][1]
        nodenumber = node_x * 64 + node_y
        if not nodenumber in input_node_map.keys():
            input_node_map[nodenumber] = {}
        input_node_map[nodenumber].update({dst % neuron_num: dst})
    gen_in.change_format(in_conv1)

    time1 = time.time()
    print(time1)

    # inputlist1, rowlist1 = gen_in.gen_inputdata(new_con, t1, input_node_map, int(interval),
    #                     'input1.txt', 'row1.txt')

    # inputlist1, rowlist1 = gen_in.gen_inputdata_list(new_con, t1, input_node_map, int(interval))
    gen_in.gen_inputdata(in_conv1, t1, input_node_map, int(interval), 'config/input.txt', 'config/row.txt')

    # inputlist1, rowlist1 = gen_in.gen_inputdata_list(in_conv1, t1, input_node_map, int(interval))

    # print(len(inputlist1))
    print('input done')

    time2 = time.time()
    print(time2 - time1)

if __name__ == "__main__":
    create_config()