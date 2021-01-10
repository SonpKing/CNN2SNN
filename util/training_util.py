import os, time, math
import torch
import pickle
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension

def read_data(file):
    with open(file, 'rb') as fo:
        items = pickle.load(fo, encoding='bytes')
    images = items[b'data']
    labels = items[b'fine_labels']
    return images, labels

def read_label_name(file):
    with open(file, 'rb') as fo:
        items = pickle.load(fo, encoding='bytes')[b'fine_label_names']
    for i in range(len(items)):
        items[i] = str(items[i], encoding = "utf-8") 
    return items 

def data_loader(root, img_size=None, batch_size=256, workers=2, pin_memory=True, dataset="imagenet"):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if dataset == "imagenet":
        if not img_size:
            img_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ])
        )
        print(train_dataset.class_to_idx)
    elif dataset == "cifar":
        if not img_size:
            img_size = 32
        train_dataset = MyDataset(file_path=train_dir, transforms=transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),#transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),#
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]))
        val_dataset = MyDataset(file_path=val_dir, transforms=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]))
    elif dataset == "imagenet_caffe":
        mean = torch.as_tensor([103.939, 116.779, 123.68])[:, None, None]#[102.9801, 115.9465, 122.7717]
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda x:torch.ByteTensor(torch.ByteStorage.from_buffer(x.tobytes())).view(img_size, img_size, 3).permute((2, 0, 1)).contiguous().float()[(2,1,0),:,:].sub_(mean))
            ])
        )
        val_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.Lambda(lambda x:torch.ByteTensor(torch.ByteStorage.from_buffer(x.tobytes())).view(img_size, img_size, 3).permute((2, 0, 1)).contiguous().float()[(2,1,0),:,:].sub_(mean))
            ])
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader


def data_loader_darwin(root, img_size, batch_size=256, workers=2, pin_memory=True, dataset="imagenet"):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            # transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # normalize,
        ])
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # normalize,
        ])
    )
    print(train_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader


class MyDataset(Dataset):
    def __init__(self, file_path = None, transforms=None, images=None, labels=None):
        if isinstance(file_path, str):
            self.images, self.labels = read_data(file_path)
        else:
            self.images = images
            self.labels = labels
        self.transforms=transforms

    def __getitem__(self, index):
        img = self.images[index].reshape(3, 32, 32).transpose((1,2,0))
        return (self.transforms(Image.fromarray(img)), self.labels[index])

    def __len__(self):
        return len(self.labels)

def data_loader_anno(root, img_size, batch_size=256, workers=2, pin_memory=True, dataset="imagenet"):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    train_dataset = MyDatasetAnno(train_dir, transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]))
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]))
    print(train_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    try:
                        conf = float(fname.split('_')[-2])
                        assert(0.0 <= conf <= 1.0)
                    except Exception as e:
                        print("conf error", fname)
                        raise e
                    item = path, class_index, conf
                    instances.append(item)
    return instances

class MyDatasetAnno(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, 
        extensions=IMG_EXTENSIONS, is_valid_file=None):
        super().__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:w
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, conf = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, conf

    
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.samples)


class MyFolderAnno(MyDatasetAnno):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count


def accuracy(output, label, top=(1,), class_acc=None):
    maxk = max(top)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    if class_acc:
        for i in range(len(label)):
            class_acc[label[i]] += 1 if correct[0, i] else 0


    res = []
    for k in top:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def read_labels(path, name=2):
    with open(path, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
    data = data.split()
    res = dict()
    for i in range(len(data) // 4):
        res[int(data[i*4])] = data[i*4+name]
    return res


def writeToTxt(list_name, file_path):
    try:
        fp = open(file_path, "w+")
        for item in list_name:
            fp.write(str(item) + "\n")
        fp.close()
    except IOError:
        print("fail to open file")
    

def convert_input_no_normalise(state_path, new_path, conv_name, device=None):
    # state_path = "/home/jinxb/.cache/torch/checkpoints/resnet18-5c106cde.pth.bak"
    # new_path = "/home/jinxb/.cache/torch/checkpoints/resnet18-5c106cde.pth"
    state = torch.load(state_path, map_location = device)['state_dict']
    miu = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    sigma = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    state[conv_name + '.bias'] -= torch.sum(torch.sum(torch.sum(miu * state[conv_name + '.weight'] / sigma, dim=-1), dim=-1), -1)
    # print(state['conv1.bias'].shape)
    state[conv_name + '.weight'] = state[conv_name + '.weight'] / sigma
    torch.save({'state_dict':state}, new_path)
    # new = torch.load(new_path)
    # print(new.keys())

def get_state(model, state_path, remove_layers, device=None, need_dict=True):
    if need_dict:
        state = torch.load(state_path, map_location = device)['state_dict']
    else:
        state = state_path
    state_multi = list(state.keys())[0].split('.')[0]=="module"
    model_multi = hasattr(model, "module")
    if model_multi!=state_multi:
        if model_multi:
            state = {"module."+name: weight for (name, weight) in state.items()}
        else:
            state = {name.replace('module.',''): weight for (name, weight) in state.items()}
    if len(remove_layers) > 0:#["block.1", "block.5", "block.6"]
        all_names = list(state.keys())
        for name in all_names:
            for layer in remove_layers:
                if layer in name:
                    state.pop(name)
                    print("remove layer", name)
    return state

def rename_layers(state_path, name_mapping, state_dict = True):
    if state_dict:
        state = torch.load(state_path)['state_dict']
    else:
        state = torch.load(state_path)
    all_state = state.items()
    state = {}
    for name, weight in all_state:
        if name in name_mapping:
            print("rename", name, "with", name_mapping[name])
            state[name_mapping[name]] = weight
        else:
            state[name] = weight
    return state

def load_pretrained(model, state_path, remove_layers=[], start_epoch=0, schedualer=None, device=None):
    print("load weights from {}".format(state_path))
    if schedualer:
        for _ in range(1, min(start_epoch, schedualer.get_steps())):
            schedualer.schedual()
    model.load_state_dict(get_state(model, state_path, remove_layers, device), strict=False)

def load_pruned(model, state_path, start_epoch, schedualer=None, device=None):
    if schedualer:
        for _ in range(1, min(start_epoch, schedualer.get_steps())):
            schedualer.schedual()
    model.load_state_dict(torch.load(state_path, map_location=device)['state_dict'])
    print("resume {} epoch from {}".format(start_epoch, state_path))


def rename(layer_name, rename_layers):
    for rname in rename_layers:
        if rname in layer_name:
            layer_name = layer_name.replace(rname, rename_layers[rname])
            break
    return layer_name
    

def load_single_to_multi(model, state_path, repeats=[], remove_layers=[], rename_layers={}):
    print("load weights from {}".format(state_path))
    old_state = get_state(model, state_path, remove_layers)
    state = dict()
    model_state = model.state_dict()
    for name in model_state:
        _name = rename(name, rename_layers)
        if _name not in old_state:
            print(_name)
            print(name, "\t\tnot exist")
            continue
        state[name] = old_state[_name]
        dim = len( list(state[name].shape) )
        if dim <= 0:
            continue
        if state[name].shape== model_state[name].shape:
            print(name, "\t\talready match")
        else:
            for repeat in repeats:
                if state[name].shape[0]*repeat == model_state[name].shape[0]:
                    shape = [1] * dim
                    shape[0] =  repeat
                    state[name] = state[name].repeat(*shape)
                    print(name, "\t\trepeat {}".format(shape[0]))
                    break
            if state[name].shape != model_state[name].shape:
                print(name, "\t\tnot match")
                state.pop(name) 
    model.load_state_dict(state, strict=False)


def load_single_to_multi2(model, state_path, state_path2, repeats=[], remove_layers=[], rename_layers={}):
    print("load weights from {}".format(state_path))
    old_state = get_state(model, state_path, remove_layers)
    old_state2 = get_state(model, state_path2, remove_layers)
    state = dict()
    state2 = dict()
    model_state = model.state_dict()
    for name in model_state:
        _name = rename(name, rename_layers)
        if _name not in old_state:
            print(name, "\t\tnot exist")
            continue
        state[name] = old_state[_name]
        state2[name] = old_state2[_name]
        dim = len( list(state[name].shape) )
        if dim <= 0:
            continue
        if state[name].shape== model_state[name].shape:
            print(name, "\t\talready match")
        else:
            for repeat in repeats:
                if isinstance(repeat, list):
                    if state[name].shape[0]*(repeat[0] + repeat[1]) == model_state[name].shape[0]:
                        shape = [1] * dim
                        shape[0] =  repeat[0]
                        shape2 = [1] * dim
                        shape2[0] =  repeat[1]
                        state[name] = torch.cat((state[name].repeat(*shape), state2[name].repeat(*shape2)), 1)
                        print(name, "\t\trepeat {}  and  {}".format(shape[0], shape2[0]))
                        break
                elif state[name].shape[0]*repeat == model_state[name].shape[0]:
                    shape = [1] * dim
                    shape[0] =  repeat
                    state[name] = state[name].repeat(*shape)
                    print(name, "\t\trepeat {}".format(shape[0]))
                    break
                
            if state[name].shape != model_state[name].shape:
                print(name, "\t\tnot match")
                state.pop(name) 
    model.load_state_dict(state, strict=False)


if __name__ == "__main__": 
    # os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
    # print(torch.cuda.device_count())
    # print(time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
    # train_loader, val_loader = gpu_data_loader(root="/home/jinxiaobo/data/imagenet", batch_size=1024, workers=2, img_size=80)
    # print("start iterate")
    # start = time.time()
    # print(len(train_loader), len(val_loader))
    # for i, (images, labels) in enumerate(train_loader):
    #     images = images.cuda(non_blocking=True)
    #     labels = labels.cuda(non_blocking=True)
    #     print(torch.max(images), torch.min(labels))
    #     print(i, labels.shape[0])
    # end = time.time()
    # print("end iterate")
    # print("dali iterate time: {0}s".format(end - start))
    # # train_loader, val_loader = data_loader(root="/home/jinxiaobo/data/imagenet", batch_size=1, workers=2, img_size=80)
    # # for it, (inputs, labels) in enumerate(train_loader):
    # #     print(inputs.shape)
    # #     print(labels.shape)


    # from backbone.MoSliceNet import moslicenet, moslicenetv3
    # model = moslicenetv3()
    # print(model)
    # rename_layers={"5.2":"5.1","5.3":"5.2","5.4":"5.3","5.5":"5.4","6.0":"5.0",
    # "6.2":"5.1","6.3":"5.2", "7.0":"5.0","7.2":"5.1","7.3":"5.2", "8.0":"6.0"}
    # load_single_to_multi(model, "checkpoint/0_pretrianed/epoch_18.pth.tar", repeats=[4*4, 8, 2], rename_layers=rename_layers)
    # # a = torch.range(0, 11).reshape(3, 2, 2)
    # # print(a.repeat(2, 1, 1))

    # read_labels("imagenet_label.txt")
    convert_input_no_normalise("checkpoint/0_pretrained_128/molicenet_max_act_no_prune_0.99999.pth.tar", "checkpoint/0_pretrained_128/molicenet_max_act_no_prune_no_normalise_0.99999.pth.tar", "conv_stem", torch.device("cpu"))

