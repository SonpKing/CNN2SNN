import torch
import torch.nn as nn
from convert import is_contain_name
import torch.nn.utils.prune as prune

def find_layers(net, all_layers):
    for (name, layers) in net.named_children():
        if name=="downsample":
            continue
        if isinstance(layers, nn.Conv2d):
            all_layers.append(layers)
        else:
            find_layers(layers, all_layers)


def parameter_to_prune(net):
    all_layers = []
    find_layers(net, all_layers)
    res = []
    for layer in all_layers:
        res.append((layer, "weight"))
    return res

def get_parameter_to_prune(net, module_names, except_names):
    res = []
    for name, module in net.named_modules():
        if is_contain_name(name, module_names) and not is_contain_name(name, except_names):
            res.append((name, module))
    return res


class PruneSchedualer:
    def __init__(self, params, prune_amount, prune_epochs, prune_iters, need_forzen, retrain_iter=1, learning_rate=0.01, retrain_lr=None):
        self.prune_params = []
        for i in range(len(prune_epochs)-1):
            for k in range(prune_epochs[i+1], prune_epochs[i]):
                iter_amount = 1 - prune_amount[k]**(1.0/prune_iters[i])
                self.prune_params.append((params[k][0], params[k][1], iter_amount))
        self.prune_params = self.prune_params[::-1]
        if isinstance(retrain_iter, int):
            retrain_iter = [retrain_iter] * len(prune_iters)

        total_iter = 0
        for i in prune_iters:
            total_iter += i
        self.epochs = prune_epochs
        self.iters = []
        self.step = -1
        for i in range(len(prune_iters)):
            for j in range(prune_iters[i]):
                self.iters.append(i+1)
                for k in range(retrain_iter[i]):
                    self.iters.append(-1)
        self.lr = learning_rate
        self.retrain_lr = retrain_lr
        self.need_forzen = need_forzen

                
    def schedual(self):
        self.step += 1
        if self.iters[self.step] != -1:
            epoch = self.iters[self.step]
            print("pruning part", epoch)
            for k in range(self.epochs[epoch], self.epochs[epoch-1]):
                # print(epoch, k, self.epochs[epoch], self.epochs[epoch-1])
                name, module, amount= self.prune_params[k]
                prune.l1_unstructured(module, "weight", amount)
                print("prune", name, "{:.3f}".format(amount), ", current sparity:", self.get_sparsity(module))
        

    def get_steps(self):
        return len(self.iters)

    def get_sparsity(self, module):
        return "{:.2f}%".format(100 - 100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement()))

    def get_lr(self):
        if self.iters[self.step] == -1:
            if not self.retrain_lr:
                return self.lr * 0.1
            else:
                return self.retrain_lr
        else:
            return self.lr

    def get_need_forzen(self):
        t = self.step
        while self.iters[t] == -1:
            t -= 1
        return self.need_forzen[self.iters[t]-1]




if __name__ == "__main__":
    # from backbone.MoSliceNet import moslicenetv10
    # net = moslicenetv10()
    # prune_amount = [0.6, 0.8] * 3 + [0.6, 0.8] * 5 + [0.6, 0.8] * 6 + [0.6, 0.7] * 5 + \
    #                 [0.6, 0.4] + [0.2, 0.25]*4 + [0.25, 0.15] + [0.15] + [0.7]
    # i = 0
    # for name, module in get_parameter_to_prune(net, ["conv", "classifier"], ["dw", "stem", "blocks.0"]):
    #     print(module.weight.shape, module.stride, module.padding, module.kernel_size, module.groups)
    #     print(name, prune_amount[i])
    #     i += 1

    pass
    # import torch.nn.utils.prune as prune
    # import torch

    # class LeNet(nn.Module):
    #     def __init__(self):
    #         super(LeNet, self).__init__()
    #         # 1 input image channel, 6 output channels, 3x3 square conv kernel
    #         self.conv1 = nn.Conv2d(100, 100, 3)
    #         self.conv2 = nn.Conv2d(6, 16, 3)
    #         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
    #         self.fc2 = nn.Linear(120, 84)
    #         self.fc3 = nn.Linear(84, 10)

    #     def forward(self, x):
    #         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    #         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    #         x = x.view(-1, int(x.nelement() / x.shape[0]))
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x

    # model = LeNet()
    # module = model.conv1
    # p = 0.2
    # amount  = 1 - p**(0.2)
    # prune.l1_unstructured(module, name="weight", amount=amount)
    # # print(list(module.named_buffers()))
    # prune.l1_unstructured(module, name="weight", amount=amount)
    # prune.l1_unstructured(module, name="weight", amount=amount)
    # prune.l1_unstructured(module, name="weight", amount=amount)
    # prune.l1_unstructured(module, name="weight", amount=amount)
    # # print(list(module.named_buffers()))
    # # print(module.weight)
    # print(
    # "Sparsity in conv1.weight: {:.2f}%".format(
    #     100. * float(torch.sum(model.conv1.weight == 0))
    #     / float(model.conv1.weight.nelement())
    # ))


    from backbone.MoSliceNet import *
    prune_amount = [0.6, 0.8] * 3 + [0.6, 0.8] * 5 + [0.6, 0.8] * 6 + [0.6, 0.7] * 5 + \
                [0.6, 0.4] + [0.25, 0.25]*4 + [0.25, 0.2, 0.2, 0.7]
    #6+10+12+10+2+8+4=52
    prune_epoch = [52, 51, 50, 48, 38, 28, 16, 6, 0]
    iters = [1, 1, 1, 1, 1, 1, 1, 1]
    retrain = [1, 5, 5, 5, 1, 1, 1, 1]
    model = moslicenetv10()
    params = get_parameter_to_prune(model, ["conv", "classifier"], ["dw", "stem", "blocks.0"])
    need_forzen = [["classifier"], ["conv_head", "module.bn2"], ["blocks.6"], ["blocks.5"], ["blocks.4"], ["blocks.3"],["blocks.2"], ["blocks.1"]]
    sch = PruneSchedualer(params, prune_amount, prune_epoch, iters, need_forzen, retrain)
    
    for i in range(sch.get_steps()):
        sch.schedual()
        print(sch.get_need_forzen())
        print(sch.get_lr())
    print(sch.get_steps())
        