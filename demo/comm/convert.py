import struct
import numpy as np

class Convert(object):
    def __init__(self):
        pass
    def convert(self,config_list):
        length = len(config_list)
        send_bytes = bytearray()
        for i in range(length):
            send_bytes += struct.pack('<Q', int(config_list[i].strip(), 16))
        return send_bytes

    def getResult(self,result):
        output_list = []
        spike = [0] * 10
       
        fmt = 'Q' * int(len(result) / 8)
        output_list += struct.unpack(fmt, result)
        print('--> output_list len', len(output_list))
        i = 0
        print(output_list)
        while(i < len(output_list)):
            leng = output_list[i]
            # print("leng is ",leng)
            i+=1
            for j in range(0, leng, 2):
                temp = int(output_list[i+j+1] % (1 << 32) // (1 << 16))
                # print(temp)
                spike[temp] += 1
            i+=leng
        
        res = np.argmax(np.asarray(spike))
        return res