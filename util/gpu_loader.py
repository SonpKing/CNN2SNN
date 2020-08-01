import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
import math 


# def reorg(x):  #(1, 3, 128, 128) -> (1, 12, 64, 64)
#     batch_size, channels, height, width = x.shape
#     _height, _width = height //2, width //2

#     x = x.view(batch_size, channels, _height, stride, _width, stride).transpose(3, 4).contiguous()
#     x = x.view(batch_size, channels, _height * _width, stride * stride).transpose(2, 3).contiguous()
#     x = x.view(batch_size, channels, stride * stride, _height, _width).transpose(1, 2).contiguous()
#     x = x.view(batch_size, -1, _height, _width)

#     return x

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1, normalise=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, exec_async=False,
                                             exec_pipelined=False)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)  #convert from cpu to gpu
        # self.res = ops.Resize(device="gpu", resize_shorter=crop, interp_type=types.INTERP_TRIANGULAR)
        # self.pad = ops.Pad(device="gpu", axes = (0, 1), shape=(224, 224), fill_value=0)#
        self.res = ops.RandomResizedCrop(device="gpu", size=(crop, crop), interp_type=types.INTERP_TRIANGULAR)  #random resize and crop the images , , 
        if normalise:
            mean = [0.485*255, 0.456*255, 0.406*255]
            std = [0.229*255, 0.224*255, 0.225*255]
        else:
            mean = [0.0, 0.0, 0.0]#normalize the value to [0, 1],#
            std = [255.0, 255.0, 255.0]
        self.cmnp = ops.CropMirrorNormalize(device="gpu",  #not providing any crop argument will result into mirroring and normalization only 
                                            # crop=[crop, crop],
                                            output_dtype=types.FLOAT,  
                                            output_layout=types.NCHW,  #shuffle the channels
                                            image_type=types.RGB,
                                            mean=mean,  
                                            std=std)
        self.coin = ops.CoinFlip(probability=0.5)
        # self.reorg = ops.PythonFunction(reorg, device="gpu")
        print('DALI "{0}" variant'.format(dali_device))
 
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        # images = self.pad(images)
        images = self.res(images)
        rng = self.coin()  #generate 1 or 0 with the probability
        output = self.cmnp(images, mirror=rng)  #mirror means horizontal flip
        # output = self.reorg(output)
        return [output, self.labels]
 
 
class HybridValPipe(Pipeline):  #Don't modify , the test result in imagenet_val is very good
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, local_rank=0, world_size=1, normalise=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, exec_async=False,
                                             exec_pipelined=False)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=crop, interp_type=types.INTERP_TRIANGULAR)  #resize
        if normalise:
            mean = [0.485*255, 0.456*255, 0.406*255]
            std = [0.229*255, 0.224*255, 0.225*255]
        else:
            mean = [0.0, 0.0, 0.0]#normalize the value to [0, 1],#
            std = [255.0, 255.0, 255.0]
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            crop=[crop, crop],
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=mean,  
                                            std=std) 
        # self.reorg = ops.PythonFunction(reorg, device="gpu")
 
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        # output = self.reorg(output)
        return [output, self.labels]
 
class MyClassificationIterator(DALIClassificationIterator):
    def __init__(self, pipeline, size):
        super(MyClassificationIterator, self).__init__(pipeline, size, auto_reset=True)

    def __len__(self):
        return math.ceil(self._size/self.batch_size)

    def __next__(self):
        # if self._first_batch is not None:  #sucking the bug in DALIGenericIterator, digusting!!!
        #     batch = self._first_batch  #however, I have remove the bug
        #     self._first_batch = None
        #     return batch
        data = super(MyClassificationIterator, self).__next__()
        return (data[0]["data"], data[0]["label"].squeeze().long())


    
 
def gpu_data_loader(root, batch_size, workers, img_size, num_gpus=1, device_id=0, world_size=1, local_rank=0, normalise=False):
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=workers, device_id=device_id,
                                    data_dir=root + '/train',
                                    crop=img_size, world_size=world_size, local_rank=local_rank, normalise=normalise)
        pip_train.build()
        dali_iter_train = MyClassificationIterator(pip_train, pip_train.epoch_size("Reader") // world_size)

        pip_val = HybridValPipe(batch_size=batch_size//8, num_threads=workers, device_id=device_id,
                                data_dir=root + '/val',
                                crop=img_size, world_size=world_size, local_rank=local_rank, normalise=normalise)
        pip_val.build()
        dali_iter_val = MyClassificationIterator(pip_val, pip_val.epoch_size("Reader") // world_size)

        return dali_iter_train, dali_iter_val