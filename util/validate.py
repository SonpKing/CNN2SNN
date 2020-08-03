import argparse
import os, time
import torch
import torchvision.transforms as transforms
from util import AverageMeter, accuracy
from models import reset_spikenet
from torch.autograd import Variable
import pickle


def validate(val_loader, model, sim_iter=None, show=True):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    max_rate = 0
    for it, (inputs, targets) in enumerate(val_loader):
        start_time = time.time()
        with torch.no_grad():
            input_var = Variable(inputs).cuda()
            target_var = Variable(targets).cuda()
            #compute

            if isinstance(sim_iter, int):
                reset_spikenet(model)
                output = None
                for _ in range(sim_iter):  #sim_iter used for spiking simulate not for ANN 
                    if isinstance(output, torch.Tensor):
                        output += model(input_var)
                    else:
                        output = model(input_var) 
                # for i in range(sim_iter):  #sim_iter used for spiking simulate not for ANN 
                #     # print("simulate", i)
                #     model(input_var)
                # output = model.act4.total_num #+ model.act4.membrane
                #     # print((output+model.if3.membrane)[0])
                #     # print(torch.argmax(output+model.if3.membrane,dim=1))
                # max_rate = max(max_rate, torch.max(output).item())
                # print(max_rate)
                # output += torch.nn.functional.softmax(model.if3.membrane, dim=1)
                # print(model.if3.membrane.shape);input()
            else: 
                output = model(input_var)
                # print(torch.argmax(output,dim=1))
            if show:
                print(output)
            # print(torch.max(output, dim=1), torch.argmax(output, dim=1), targets)

            # print(target_var)
            # input()
            loss = criterion(output, target_var)

            #measure
            prec = accuracy(output.data, target_var.data, top=(1,))
            num_batch = targets.size(0)
            losses.update(loss.data.item(), num_batch)
            top1.update(prec[0].item(), num_batch)
        batch_time.update(time.time()-start_time)
        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' #'Data {data_time.val:.3f} ({data_time.avg():.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
            batch_time=batch_time, loss=losses, top1=top1))


    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def eval_single(inputs, model, sim_iter=None):
    with torch.no_grad():
        reset_spikenet(model)
        input_var = Variable(inputs)
        output = None
        for _ in range(sim_iter):
            if isinstance(output, torch.Tensor):
                output += model(input_var)
            else:
                output = model(input_var) 
    return output


if __name__ == "__main__":
    pass
