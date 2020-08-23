class neuron(object):
    def __init__(self, conns={}):  # conns is a map of {nodenumbers : [(neuron_id<<8)+weight,...],... }
        self.conns = conns

    def set_conn(self, node_number, conn):
        self.conns[node_number] = conn


class node(object):
    # address unit is word = 4 Bytes
    def __init__(self, node_number, zero_ref, neurons, delay=[0, 0], vth=16, leak=0, reset=0, leaksign=0, grid_size=64,
                 npu_r=0, npu_m=1, ni_r=2):
        import math
        self.npu_r = npu_r
        self.npu_m = npu_m
        self.ni_r = ni_r
        self.grid_size = grid_size
        self.body_pack_head = ((24 - 1) << 38) | (0b1 << 32) ####change
        self.choosebits = math.ceil(math.log(grid_size, 2))
        self.delay = delay
        self.vth = vth
        self.leak = leak
        self.reset = reset
        self.leaksign = leaksign
        self.set_nodenum(node_number)
        self.set_zeroref(zero_ref)
        self.set_neurons(neurons)

    def set_nodenum(self, node_number):
        self.x = node_number // self.grid_size
        self.y = node_number % self.grid_size

    def set_zeroref(self, zero_ref):
        self.zero_ref = zero_ref  # 2 elements' tuple

    def set_neurons(self, neurons):
        self.neurons = neurons
        self.neuron_num = len(neurons)
        self.linker_baddr = 0
        self.pack_baddr = self.neuron_num + 1

    def cal_addr(self):
        linker = []
        data = []  # addr based on self.pack_baddr
        num = 0
        word_count = 0
        for neuron in self.neurons:
            linker.append([num, word_count])
            for key, v in neuron.conns.items():  # head: id,weight, x,y,pack_size   tail: 0,0,id,weight
                flag = 0
                x = key // self.grid_size - self.zero_ref[0]
                y = key % self.grid_size - self.zero_ref[1]
                pack_size = len(v)  # 1 <= pack_size <= 255
                head = (((x << 4) | y) << 8) | pack_size
                for connection in v:
                    if flag == 0:
                        flag = 1
                        data.append([word_count + self.pack_baddr, ((connection << 16) | head)])
                        word_count += 1
                    else:
                        flag = 0
                        head = connection
                if flag == 0:
                    data.append([word_count + self.pack_baddr, head])
                    word_count += 1
            num += 1
        linker.append([num, word_count])
        return linker, data

    def get_nodenum(self):
        return self.x, self.y

    def get_zeroref(self):
        return self.zero_ref

    def get_packlength(self):
        return self.choosebits + 38

    def get_headpack(self):  # from (grid_size,grid_size) to (x,y)       dst_port: 0 -> 1,  2 -> 2, 3 -> 4
        #tmp = (self.x << 18) | (self.y << 12) | (self.grid_size << 6) | (self.grid_size - 1)
        tmp = (self.x << 18) | (self.y << 12) | (48 << 6) | (47)  #  input 点s设为(49,49)
        x1 = self.x % 24
        y1 = self.y % 24
        if x1 == 23:
            return (y1 << 38) | (0b10 << 36) | (0b1 << 32) | (0b1 << 29) | (0x1 << 24) | tmp
        else:
            return (y1<< 38) | (0b10 << 36) | (0b1 << 32) | (0b1 << 29) | (0x4 << 24) | tmp
        # if self.x == self.grid_size - 1:
        #     return (self.y << 38) | (0b10 << 36) | (0b1 << 32) | (0b1 << 29) | (0x1 << 24) | tmp
        # else:
        #     return (self.y << 38) | (0b10 << 36) | (0b1 << 32) | (0b1 << 29) | (0x4 << 24) | tmp

    def get_bodypackhead(self):
        y1 = self.y % 24
        return (y1 << 38) | (0b1 << 32)
        # return (self.y << 38) | (0b1 << 32)

    def get_tailpackhead(self):
        y1 = self.y % 24
        return (y1 << 38) | (0b01 << 36) | (0b1 << 32)
        # return (self.y << 38) | (0b01 << 36) | (0b1 << 32)

    """
    [31:24]		
            31:     1'b0: Read command.
                    1'b1: Write command.
        [30:29]:    2'b00: Command to NPU registers.
                    2'b01: Command to NPU memory.
                    2'b10: Command to NI.
                    2'b11: Command to Configurator.
    Other:  Preserved.
    [23:0]		Address of the command.
    """

    def setr_neuronnum(self, rw=1, v=-1):  # 0x0c
        """
        [31:8]: Reserved.
        [7:0]: The number of on-chip neurons. Note that this register has only 8 bits so that its value indicates the number of on-chip neurons minus one.
        """
        if v == -1:
            v = self.neuron_num
        res = []
        assert rw == 1 or rw == 0
        comm = (rw << 31) | (self.npu_r << 29) | 0x0c
        res.append(comm)
        if rw == 1:
            res.append(v - 1)
        return res

    def setr_leak(self, rw=1, v=-1):  # 0x10
        """
        [31:0]: Neuromorphic computing parameter leak.
        """
        if v == -1:
            v = self.leak
        res = []
        assert rw == 1 or rw == 0
        comm = (rw << 31) | (self.npu_r << 29) | 0x10
        res.append(comm)
        if rw == 1:
            res.append(v)
        return res

    def setr_vth(self, rw=1, v=-1):  # 0x14
        """
        [31:0]: Neuromorphic computing parameter of vth.
        """
        if v == -1:
            v = self.vth
        res = []
        assert rw == 1 or rw == 0
        comm = (rw << 31) | (self.npu_r << 29) | 0x14
        res.append(comm)
        if rw == 1:
            res.append(v)
        return res

    def setr_mode(self, rw=1, v=-1, leaksign=0):  # 0x18
        """
        This register holds the configuration of extra neuromorphic computing settings.
        [31:2]: Reserved.
        [5:4]: leak mode.
            00: zero.
            01: +
            10: -
        [0]: Reset mode.
            0: Reset to zero.
            1: Reset by subtraction.
        """
        if v == -1:
            v = 0
        res = []
        assert rw == 1 or rw == 0
        comm = (rw << 31) | (self.npu_r << 29) | 0x18
        res.append(comm)
        # if rw==1:
        #     if leaksign == 0:
        #         res.append(v)
        #     elif leaksign == -1:
        #         res.append(1<<5+v)
        #     else:
        #         res.append(1<<4+v)

        if rw == 1:
            if leaksign == 0:
                com_data = v
            elif leaksign == -1:
                com_data = (1 << 5 + v)
            else:
                com_data = (1 << 4 + v)
        com_data += (self.reset)
        res.append(com_data)

        return res

    def setr_packet_settings(self, rw=1, v=-1):  # 0x1c
        """
        This register holds the configuration of extra packet settings.
        [31:1]: Reserved.
        [0]: Quantization mode.
            0: Linear quantization.
            1: Non-linear quantization.
        """
        pass

    def setr_linker_baddr(self, rw=1, v=-1):  # 0x40
        """
        [31:17]: Reserved.
        [16:0]: Linker base address.
        """
        if v == -1:
            v = self.linker_baddr
        res = []
        assert rw == 1 or rw == 0
        comm = (rw << 31) | (self.npu_r << 29) | 0x40
        res.append(comm)
        if rw == 1:
            res.append(v)
        return res

    def setr_packet_baddr(self, rw=1, v=-1):  # 0x48
        """
        [31:17]: Reserved.
        [16:0]: Packet base address.

        """
        if v == -1:
            v = self.pack_baddr
        res = []
        assert rw == 1 or rw == 0
        comm = (rw << 31) | (self.npu_r << 29) | 0x48
        res.append(comm)
        if rw == 1:
            res.append(v)
        return res

    def setr_status(self, rw=1, v=2):  # 0x00  clear by default
        """
        1: Clear bit.
        0: Enable bit.
        """
        assert v != -1 and rw == 1
        res = []
        comm = (rw << 31) | (self.npu_r << 29) | 0x0
        res.append(comm)
        res.append(v)
        return res

    def setr_nizeroref(self, rw=1, v=-1):
        if v == -1:
            v = self.zero_ref
        res = []
        assert rw == 1 or rw == 0
        comm = (rw << 31) | (self.ni_r << 29) | 0x1
        res.append(comm)
        if rw == 1:
            x = v[0]
            y = v[1]
            v = (x << 8) | y
            res.append(v)
        return res

    def setr_nidelay(self, rw=1, v=-1):
        if v == -1:
            v = self.delay
        res = []
        assert rw == 1 or rw == 0
        comm = (rw << 31) | (self.ni_r << 29) | 0x2
        res.append(comm)
        if rw == 1:
            res.append(v[0])
        comm = (rw << 31) | (self.ni_r << 29) | 0x3
        res.append(comm)
        if rw == 1:
            res.append(v[1])
        return res


class configuration(object):
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.mmc = 1
        self.enable = 1

    def gen_config_file(self, filename, leaksign=0):
        f = open(filename, 'w')
        rf = open("re_config.txt", "w")
        # f.write(str+'\n')
        for node in self.nodes:
            """
            set registers and memory
            """
            #print(node.get_packlength())
            #print(node,node.x,node.y,node.zero_ref)
            tmp = node.get_headpack()
            #print("%011x" % tmp)
            body_pack_head = node.get_bodypackhead()
            tail_pack_head = node.get_tailpackhead()
            con_head = '40000'
            ss = "%011x" % tmp  # head
            f.write(con_head + ss + '\n')
            rf.write(con_head + ss + '\n')
            # set neunum first and then clear
            tmp = node.setr_neuronnum()
            for t in tmp:
                ss = "%011x" % (t + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
            tmp = node.setr_status()
            for t in tmp:
                ss = "%011x" % (t + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
            tmp = node.setr_vth()
            for t in tmp:
                ss = "%011x" % (t + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
            tmp = node.setr_leak()
            for t in tmp:
                ss = "%011x" % (t + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
            tmp = node.setr_mode(leaksign=leaksign)
            for t in tmp:
                ss = "%011x" % (t + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
            tmp = node.setr_linker_baddr()
            for t in tmp:
                ss = "%011x" % (t + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
            tmp = node.setr_packet_baddr()
            for t in tmp:
                ss = "%011x" % (t + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
            tmp = node.setr_nizeroref()
            for t in tmp:
                ss = "%011x" % (t + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
            tmp = node.setr_nidelay()
            for index, t in enumerate(tmp):
                if index == len(tmp) - 1 and not self.mmc:
                    ss = "%011x" % (t + tail_pack_head)
                else:
                    ss = "%011x" % (t + body_pack_head)
                    
                f.write(con_head + ss + '\n')
            for index, t in enumerate(tmp):
                if index == len(tmp) - 1 :
                    ss = "%011x" % (t + tail_pack_head)
                else:
                    ss = "%011x" % (t + body_pack_head)
                rf.write(con_head + ss + '\n')
            if self.mmc:
                linker, data = node.cal_addr()
                for d in linker:
                    h = (1 << 31) | (node.npu_m << 29) | d[0]
                    ss = "%011x" % (h + body_pack_head)
                    f.write(con_head + ss + '\n')
                    ss = "%011x" % (d[1] + body_pack_head)
                    f.write(con_head + ss + '\n')
                for index, d in enumerate(data):
                    h = (1 << 31) | (node.npu_m << 29) | d[0]
                    ss = "%011x" % (h + body_pack_head)
                    f.write(con_head + ss + '\n')
                    if index == len(data) - 1:
                        ss = "%011x" % (d[1] + tail_pack_head)
                    else:
                        ss = "%011x" % (d[1] + body_pack_head)
                    f.write(con_head + ss + '\n')
            # tmp=node.setr_nizeroref()
            # ss="%011x" % (tmp[0]+body_pack_head)
            # f.write(ss+'\n')
            # ss="%011x" % (tmp[1]+tail_pack_head)
            # f.write(ss+'\n')
        if self.enable:
            for node in self.nodes:
                #print("aaaaaaaaaaaaa")
                tmp = node.get_headpack()
                body_pack_head = node.get_bodypackhead()
                tail_pack_head = node.get_tailpackhead()
                ss = "%011x" % tmp  # head
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
                tmp = node.setr_status(1, 1)
                ss = "%011x" % (tmp[0] + body_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
                ss = "%011x" % (tmp[1] + tail_pack_head)
                f.write(con_head + ss + '\n')
                rf.write(con_head + ss + '\n')
        f.close()


if __name__ == "__main__":
    n = neuron()
    n.set_conn(1, [1])
    ns = [n]
    no = node(0, (24, 24), ns)
    nos = [no]
    conjf = configuration(nos)
    # configuration.gen_config_file("tt.txt")