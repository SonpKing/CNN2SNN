import numpy as np
from time import sleep
from .comm import ServerListener
from multiprocessing import Manager, Pool, Process

def data2bytearray(data, dtype):
    shape = bytearray(np.array(data.shape, dtype=np.int32))
    shape = list2bytearray([len(shape)]) + shape
    data = bytearray(np.array(data, dtype=dtype))
    data = list2bytearray([len(data)]) + data
    return shape + data

def bytearray2data(bytes_data, dtype, ind=0, hook=None):
    size = np.frombuffer(bytes_data[ind: ind + 4], dtype=np.int32)[0]
    ind += 4
    shape = np.frombuffer(bytes_data[ind : ind + size], dtype=np.int32)
    ind += size
    size = np.frombuffer(bytes_data[ind: ind + 4], dtype=np.int32)[0]
    ind += 4
    data = np.frombuffer(bytes_data[ind : ind + size], dtype=dtype)
    if hook and isinstance(hook, list):
        hook[0] = ind + size
    return data.reshape(shape)

class BytearrayToDataHelper:
    def __init__(self):
        self.ind = 0

    def get_next(self, bytes_data, dtype):
        hook = [0]
        res = bytearray2data(bytes_data, dtype, self.ind, hook)
        self.ind = hook[0]
        return res

def img2bytearray(img):
    assert img.dtype == np.uint8
    shape = bytearray(np.array(img.shape, dtype=np.int32))
    print(len(shape))
    return shape + bytearray(img)

def bytearray2img(bytes_data):
    shape = np.frombuffer(bytes_data[:12], dtype=np.int32)
    img = np.frombuffer(bytes_data[12:], dtype=np.uint8)
    return img.reshape(shape)

def list2bytearray(data):
    return bytearray(np.array(data, dtype=np.int32))

def bytearray2list(bytes_data):
    return np.frombuffer(bytes_data, dtype=np.int32)

def package_to_bytearray(package_input):
    assert package_input[1].dtype == np.float64
    return data2bytearray(package_input[0], np.int32) +\
        data2bytearray(package_input[1], np.float32)

def bytearray_to_package(bytes_data):
    helper = BytearrayToDataHelper()
    inds = helper.get_next(bytes_data, np.int32)
    imgs = helper.get_next(bytes_data, np.float32)
    return inds, imgs

# def class_out_to_bytearray(inds, class_out):
#     return data2bytearray(inds, np.int32) +\
#         data2bytearray(class_out, np.int32)

# def bytearray_to_class_out(bytes_data):
#     helper = BytearrayToDataHelper()
#     inds = helper.get_next(bytes_data, np.int32)
#     outs = helper.get_next(bytes_data, np.int32)
#     return inds, outs

def class_out_to_bytearray(inds, class_out):
    return data2bytearray(inds, np.int32) +\
        data2bytearray(class_out, np.float32)

def bytearray_to_class_out(bytes_data):
    helper = BytearrayToDataHelper()
    inds = helper.get_next(bytes_data, np.int32)
    outs = helper.get_next(bytes_data, np.float32)
    return inds, outs


def recv_until_ok(conn):
    try:
        _, data = conn.receive()
        while data == None:
            sleep(0.01)
            _, data = conn.receive()
        print("recevied")
    except ServerListener.TimeoutException:
        pass
    return data

class PoolHelper:
    def __init__(self, pool_num=17):
        self.pool = Pool(processes=pool_num)

    def execute(self, func, args):
        self.pool.apply_async(func, args)

    def close(self):
        self.pool.close()
        self.pool.join()
        print("pool closed")

    def __del__(self):
        self.close()
    

def fit_input(inputs):
    assert np.max(inputs) <= 1.0 and np.min(inputs) >= 0.0
    assert inputs.shape[3] == 3
    return inputs.transpose((0, 3, 1, 2))


if __name__ == "__main__":
    a = np.array([1, 2])
    b = np.arange(48).reshape(2, 2, 3, 4)
    c = (a, b)
    bytes_data = class_out_to_bytearray(*c)
    c_data = bytearray_to_class_out(bytes_data)
    print(c_data)
