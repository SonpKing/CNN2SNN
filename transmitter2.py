import socket
import struct
import time


class Transmitter(object):
    def __init__(self):
        self.socket_inst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_inst.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def connect_lwip(self, ip_address):
        self.socket_inst.connect(ip_address)

    def close(self):
        self.socket_inst.close()

    
    def send_config(self, config_file):
        '''
        发送配置信息
        config_file： 配置信息文件名
        '''
        with open(config_file, 'r') as file:
            config_list = file.readlines()
        length = len(config_list)
        head = struct.pack("I", (1 << 28) + length)
        self.socket_inst.sendall(head)
        send_bytes = bytearray()
        for i in range(length):
            send_bytes += struct.pack('Q', int(config_list[i].strip(), 16))
        self.socket_inst.sendall(send_bytes)
        reply = self.socket_inst.recv(1024)
        reply_len = len(reply)
        if reply_len == 4:
            # print('config send done')
            return 1
        elif reply_len > 4:
            result = self.read_output(reply)
            pr = []
            for j in range(len(result)):
                pr.append(hex(result[j]))
            print(pr)
        else:
            print('config err')
            return 0
    
    def send_read_ins(self, read_file):
        '''
        发送读取命令
        read_file： 读取命令文件名
        '''
        with open(read_file, 'r') as file:
            read_list = file.readlines()
        length = len(read_list)
        head = struct.pack("I", (3 << 28) + length)
        self.socket_inst.sendall(head)
        send_bytes = bytearray()
        for i in range(length):
            send_bytes += struct.pack('Q', int(read_list[i].strip(), 16))
        self.socket_inst.sendall(send_bytes)
        
        reply = self.socket_inst.recv(1024)
        reply_len = len(reply)
        pr = []
        if reply_len == 4:
            print('read send done')
            return 1
        elif reply_len > 4:
            print('read v')
            result = self.read_output(reply)
            vth = []
            for i in range(0, len(result), 2):
                num = result[i + 1]
                if ((num >> 19) % 2 == 1) :
                    vth.append(num%(1<<19) - (1<<19))
                else:
                    vth.append(num %(1<<20))
 
            print(vth)

            # for j in range(len(result)):
            #         pr.append(hex(result[j]))
            # print(pr)
            
        else:
            print('config err')
            return 0

    def send_per_tick(self, input_list):
        '''
        单个tick发送
        input_list：input列表
        '''
        length = len(input_list)
        head = struct.pack("I", (2 << 28) + length)
        self.socket_inst.sendall(head)
        send_bytes = bytearray()
        for i in range(length):
            send_bytes += struct.pack('Q', int(input_list[i].strip(), 16))
        self.socket_inst.sendall(send_bytes)

    def send_input(self, input_spk, input_row):
        '''
        发送input文件，txt文件版
        input_spk：输入脉冲文件
        input_row：行文件
        '''
        with open(input_row, 'r') as row:
            row_list = row.readlines()
        with open(input_spk, 'r') as file:
            input_list = file.readlines()
        spike = [0] * 10
        pos = 0
        idx = 0
        
        while pos < len(input_list):
            pr = []
            rank = int(row_list[idx].strip(), 16)
            idx += 1
            input_tick = input_list[pos:rank]
            # print('----------tick---------' + str(idx - 1))
            self.send_per_tick(input_tick)
            reply = self.socket_inst.recv(1024)
            reply_len = len(reply)
            if reply_len > 4 :
                result = self.read_output(reply)
                for j in range(len(result)):
                    pr.append(hex(result[j]))
                print(pr)
                for i in range(0, len(result), 2):
                    spike[int(result[i + 1] % (1 << 32) // (1 << 16))] += 1
                
            pos = rank
            # if idx == 6:
            #     self.send_read_ins('read.txt')
           
        return spike
    
    def send_input_list(self, input_list, row_list):
        '''
        发送input文件，list版
        input_list：输入脉冲列表
        input_list：行列表
        '''
        spike = [0] * 10
        pos = 0
        idx = 0
        
        while pos < len(input_list):
            pr = []
            rank = int(row_list[idx].strip(), 16)
            idx += 1
            input_tick = input_list[pos:rank]
            # print('----------tick---------' + str(idx - 1))
            self.send_per_tick(input_tick)
            reply = self.socket_inst.recv(1024)
            reply_len = len(reply)
            if reply_len > 4:
                result = self.read_output(reply)
                for j in range(len(result)):
                    pr.append(hex(result[j]))
                # print(pr)
                xiaobo_pr = []
                for i in range(0, len(result), 2):
                    xiaobo_pr.append(int(result[i + 1] % (1 << 32) // (1 << 16)))
                    spike[int(result[i + 1] % (1 << 32) // (1 << 16))] += 1
                # print("spike:", (xiaobo_pr))
                
            pos = rank
            # if idx == 6:
            #     self.send_read_ins('read.txt')
        
           
        return spike

    def send_clear(self, clear_file):
        '''
        发送clear文件，发送结束mode不拉低
        '''
        with open(clear_file, 'r') as file:
            clear_list = file.readlines()
        length = len(clear_list)
        head = struct.pack("I", (7 << 28) + length)
        self.socket_inst.sendall(head)
        send_bytes = bytearray()
        for i in range(length):
            send_bytes += struct.pack('Q', int(clear_list[i].strip(), 16))
        self.socket_inst.sendall(send_bytes)
        reply = self.socket_inst.recv(1024)
        reply_len = len(reply)
        if reply_len == 4:
            # print('clear')
            return 1
        else:
            print('clear err')
            return 0

    def read_output(self, receive):
        '''
        读取返回
        retrun:芯片返回spk列表
        '''
        # receive = self.socket_inst.recv(1024)
        output_list = []
        fmt = 'Q' * int(len(receive) / 8)
        output_list += struct.unpack(fmt, receive)
        return output_list

    def asic_reset(self):
        '''
        芯片复位
        '''
        ins = struct.pack("I", (4 << 28))
        self.socket_inst.sendall(ins)

    def set_tick_time(self, times):
        '''
        设置tick时间
        '''
        ins = struct.pack("I", (5 << 28) + times)
        self.socket_inst.sendall(ins)

    def tick_alone(self, show=True):
        '''
        单启动tick 不发数据
        '''
        ins = struct.pack("I", (6 << 28))
        self.socket_inst.sendall(ins)
        reply = self.socket_inst.recv(1024)
        if show:
            print("tick_alone")
        spike = [0] * 10
        pr = []
        reply_len = len(reply)
        if reply_len > 4 :
            result = self.read_output(reply)
            for i in range(0, len(result), 2):
                    spike[int(result[i + 1] % (1 << 32) // (1 << 16))] += 1
            for j in range(len(result)):
                    pr.append(hex(result[j]))
            print(pr)
        return spike

    
    def auto_tick(self, length):
        '''
        发送上个tick数据
        '''
        ins = struct.pack("I", (8 << 28) + length)
        self.socket_inst.sendall(ins)
        reply = self.socket_inst.recv(1024)
        spike = [0] * 10
        pr = []
        reply_len = len(reply)
        if reply_len > 4 :
            result = self.read_output(reply)
            for j in range(len(result)):
                pr.append(hex(result[j]))
            # print(pr)
            for i in range(0, len(result), 2):
                    spike[int(result[i + 1] % (1 << 32) // (1 << 16))] += 1
        return spike       
            
       
