import math
import numpy as np

#Neu = 256.0
flag = 0

def valid(zero_node, node):
    if(node[0] < (zero_node[0] + 16) and node[1] < (zero_node[1] + 16)):
        return 1
    else:
        print("error", zero_node, node)
        return 0

def get_zero(node_next):
    xmin = float('inf')
    ymin = float('inf')
    for node in node_next:
        if node[0] < xmin:
            xmin = node[0]
        if node[1] < ymin:
            ymin = node[1]
    for node in node_next:
        if valid((xmin, ymin), node) == 0:
            print(node_next.index(node))
    return (xmin, ymin)

def map_chip11(nodes, node_link, layer_num, layer_id, forward_node):
    ###########node_num = 24
    """
    layers:list,[layerwidth1,layerwidth2,......]
    avg_conn:list,
    return: list, [(node11,node12,.....),(node21,node22,......),......]
    """
    occupy_all = np.zeros((23, 23))
    occupy_z = np.zeros((16, 23))
    empty_t = 16 * 23
    empty_b = 7 * 23
    empty_z = 16 * 12
    t_b = 1  # from top block

    nodelist = []
    zerolist = []
    forward_res = []
    first_layer = []
    ##### first layer  (输入第一层，没有来自其他的转发)
    if layer_id == 0:
        for i in range(int(nodes[layer_id])):
            x = i % 23
            y = i // 23
            occupy_all[x, y] = 1
            # empty_all -= 1
            if x < 16:
                empty_t -= 1
            else:
                empty_b -= 1
            first_layer.append((x + 25, y + 25))
        nodelist.append(first_layer)
        if layer_num == 1:
            print("layer_num == 1")
            return nodelist
    else:  ####上一块板的虚拟转发
        for node in forward_node:
            x = node[0] - 24
            y = node[1]
            occupy_all[x - 24, y - 24] = 1
            # empty_all -= 1
            if x - 24 < 16:
                empty_t -= 1
            else:
                empty_b -= 1
            first_layer.append((x, y))
        # for i in range(16):
        #     for j in range (12):
        #         occupy_all[i,j] = 1
        #         empty_t -= 1
        nodelist.append(first_layer)
    layer_id += 1
    xt0 = (16 * 23 - empty_t - 1) % 16
    yt0 = (16 * 23 - empty_t - 1) // 16
    print(empty_t, xt0, yt0)

    # if layer_id > 1:
    while layer_id < layer_num :
        if layer_id == 7:
            pass
            # break
        link = node_link[layer_id - 1]
        # print(link)
        print(layer_id, nodes[layer_id])
        curnodes = nodes[layer_id]  # nodes num of current layer
        lastnodes = nodes[layer_id - 1]
        cur_res = []  # 当前层暂定坐标
        # cur_zero = [] #与当前层有链接的上层点的原点
        last_zero = []  # 上一层原点坐标
        for i in range(curnodes):
            cur_res.append((-1, -1))

        if curnodes > empty_t + empty_b:#需转发
            break
        if t_b == 1 and empty_t >= curnodes: # 该层可都放上面
            t_b = 1
        elif t_b == 1 and empty_t > 256:  # 该层上面放不下  上面空间大于16 * 16
            t_b = 1 + 3
        elif t_b == 1 and empty_t > 128:  #### 可能需要修改128
            t_b = 2
            if xt0 != 0:
                yt0 += 1
            xt0 = 0
        elif t_b == 1 and empty_t > 0: #放弃 x [0-7]
            t_b = 2
            if empty_t < 9:
                pass
            elif xt0 < 6:
                empty_t -= 7 * (22 - yt0) + 6 - xt0
                xt0 = 7
            elif xt0 == 6:
                empty_t -= 7 * (22 - yt0)
                xt0 = 7
            else:
                empty_t -= 7 * (22 - yt0)
                yt0 += 1
                xt0 = 7
        elif t_b == 1:
            t_b = 3
        elif t_b == 2:
            t_b = 3

        for i in range(lastnodes):
            node_next = []
            i_link = link[i]
            for id in i_link:
                if cur_res[id][0] < 0:
                    if t_b % 3 == 1:
                        while occupy_all[xt0, yt0] == 1:
                            yt0 += (xt0 + 1) // 16
                            xt0 = (xt0 + 1) % 16
                        if t_b == 4 and yt0 == 16:
                            t_b = 2
                        empty_t -= 1
                    elif t_b == 2:
                        while occupy_all[xt0, yt0] == 1:
                            xt0 += (yt0 + 1) // 23
                            yt0 = (yt0 + 1) % 23
                        empty_t -= 1
                        if empty_t == 0:
                            t_b = 3
                    elif t_b == 3:
                        while occupy_all[xt0, yt0] == 1:
                            yt0 -= (xt0 - 16 + 1) // 7
                            xt0 = (xt0 - 16 + 1) % 7 + 16
                        empty_b -= 1
                    cur_res[id] = (xt0 + 25, yt0 + 25)
                    # print(xt0,yt0,cur_res[id])
                    occupy_all[xt0, yt0] = 1
                node_next.append(cur_res[id])
            zero_node = get_zero(node_next)
            last_zero.append(zero_node)

        nodelist.append(cur_res)
        zerolist.append(last_zero)
        layer_id += 1


    ###  映射已结束，输出层的zero_ref
    if layer_id == layer_num:
        last_zero = []
        for i in range(nodes[-1]):
            last_zero.append((47, 47))
        zerolist.append(last_zero)
        flag = 1
        
    ######  映射未结束，虚拟转发forward
    else:
        forwardnodes = nodes[layer_id]
        link = node_link[layer_id - 1]
        # print(link)
        # print(layer_id, nodes[layer_id])
        lastnodes = nodes[layer_id - 1]
        last_zero = []  # 上一层原点坐标
        for i in range(forwardnodes):
            forward_res.append((-1, -1))
        if forwardnodes > empty_z:
            print("can not forward")
        xt1 = 0
        yt1 = 0
        for i in range(lastnodes):
            node_next = []
            i_link = link[i]
            for id in i_link:
                if forward_res[id][0] < 0:
                    while occupy_z[xt1, yt1] == 1:
                        yt1 += (xt1 + 1) // 16
                        xt1 = (xt1 + 1) % 16
                    empty_z -= 1
                    forward_res[id] = (xt1 + 48, yt1 + 24)
                    # forward_res[id] = (49, 49)
                    # print(xt1,yt1,forward_res[id])
                    occupy_z[xt1, yt1] = 1
                node_next.append(forward_res[id])
            zero_node = get_zero(node_next)
            # zero_node = (47, 47)
            last_zero.append(zero_node)
        zerolist.append(last_zero)

    return nodelist, zerolist, layer_id, forward_res

if __name__ == "__main__":


    # nodes = [25, 16, 13, 5, 1, 1]
    nodes = [25, 21, 13, 7, 1, 1]
    # neus = [256, 77, 256, 119, 256, 256]
    # node_link = [[[0, 1], [0, 1], [1, 2], [1, 2], [2, 3], [3, 2, 4], [4, 3], [4, 5], [5, 4], [5, 6], [5, 6, 7], [7, 6], [7, 8], [8, 9], [8, 9, 10], [10, 9], [10, 11], [10, 11], [11, 12], [11, 12, 13], [13, 12], [13, 14], [14, 13], [14, 15], [15, 14]], [[0, 1], [0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12], [10, 11, 12], [11, 12]], [[0], [0], [0, 1], [0, 1], [1, 2], [1, 2], [2], [2, 3], [3, 2], [3, 4], [4, 3], [4], [4]], [[0], [0], [0], [0], [0]], [[0]]]
    # link_to = [[[0, 1], [0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10], [9, 10, 11], [10, 11, 12], [12, 13, 14], [13, 14, 15], [14, 15, 16, 17], [16, 17, 18, 19], [18, 19, 20], [19, 20, 21, 22], [21, 22, 23, 24], [23, 24]], [[0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13], [11, 12, 13, 14], [12, 13, 14, 15], [13, 14, 15]], [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10], [9, 10, 11, 12]], [[0, 1, 2, 3, 4]], [[0]]]
    node_link = [[[0, 1], [0, 1], [1, 2], [2, 3], [3, 4, 2], [3, 4, 5], [5, 4], [5, 6, 7], [7, 5, 6], [7, 8], [7, 8, 9], [9, 10, 8], [9, 10, 11], [11, 10], [11, 12, 13], [13, 11, 12], [13, 14], [13, 14, 15], [15, 16, 14], [15, 16, 17], [17, 16], [17, 18], [18, 19, 17], [19, 20], [20, 19]], [[0, 1], [0, 1, 2], [0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4], [2, 3, 4, 5], [3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9], [7, 8, 9, 10], [8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12], [10, 11, 12], [10, 11, 12], [11, 12]], [[0], [0, 1], [1], [1, 2], [2], [2, 3], [3], [3, 4], [4], [4, 5], [5], [5, 6], [6]], [[0], [0], [0], [0], [0], [0], [0]], [[0]]]

    """
    nodelist, zero_ref ,layer_id= mapping_chip(nodes, neus , 6, 0 , '00')
    nodelist1, zero_ref1, layer_id1 = mapping_chip(nodes, neus, 6, 0, '11')
    nodelist.extend(nodelist1)
    zero_ref.extend(zero_ref1)
    """
    # nodelist, zero_ref, layer_id, forward_res= mapping_board(nodes,neus,9,0,2,[])
    # forward_node = [(48, 24), (51, 24), (53, 24), (55, 24), (57, 24), (49, 24), (50, 24), (52, 24), (54, 24), (56, 24), (58, 24), (59, 24), (60, 24), (61, 24), (62, 24), (63, 24), (48, 25), (49, 25), (50, 25), (51, 25), (52, 25), (53, 25), (54, 25), (55, 25), (56, 25), (57, 25), (58, 25), (59, 25), (60, 25), (61, 25), (62, 25), (63, 25), (48, 26), (49, 26), (50, 26), (51, 26), (52, 26), (53, 26), (54, 26), (55, 26), (56, 26), (57, 26), (58, 26), (59, 26), (60, 26), (61, 26), (62, 26), (63, 26), (48, 27), (49, 27), (50, 27), (51, 27), (52, 27), (53, 27), (54, 27), (55, 27), (56, 27), (57, 27), (58, 27), (59, 27), (60, 27), (61, 27), (62, 27), (63, 27), (48, 28), (49, 28), (50, 28), (51, 28), (52, 28), (53, 28), (54, 28), (55, 28), (56, 28), (57, 28), (58, 28), (59, 28), (60, 28), (61, 28), (62, 28), (63, 28), (48, 29), (49, 29), (50, 29), (51, 29), (52, 29), (53, 29), (54, 29), (55, 29), (56, 29), (57, 29), (58, 29), (59, 29), (60, 29), (61, 29), (62, 29), (63, 29), (48, 30), (49, 30), (50, 30), (51, 30), (52, 30), (53, 30), (54, 30), (55, 30), (56, 30), (57, 30), (58, 30), (59, 30), (60, 30), (61, 30), (62, 30), (63, 30), (48, 31), (49, 31), (50, 31), (51, 31), (52, 31), (53, 31), (54, 31), (55, 31), (56, 31), (57, 31), (58, 31), (59, 31), (60, 31), (61, 31), (62, 31), (63, 31), (48, 32), (49, 32), (50, 32), (51, 32), (52, 32), (53, 32), (54, 32), (55, 32), (56, 32), (57, 32), (58, 32), (59, 32), (60, 32), (61, 32), (62, 32), (63, 32), (48, 33), (49, 33), (50, 33), (51, 33), (52, 33), (53, 33)]

    nodelist, zerolist, layer_id, forward_res = map_chip11(nodes, node_link, 6, 0, [])
    print(nodelist)
    print(zerolist)
    print(layer_id)
    print(forward_res)

