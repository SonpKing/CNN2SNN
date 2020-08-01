import argparse
import os, time, random, warnings
from util import AverageMeter, accuracy, load_pretrained, load_pruned
import torch
import torch.nn as nn
 
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.nn.utils.prune as prune

args = None
def fit(args_, model, train_loader, val_loader, learning_rate, rm_layers=[], forzen_epoch=[], need_forzen=[], schedualer=None, teacher_model=None):#, teacher_device=None, , forzen_multi_gpu=0, prune_params=[], prune_epoch=[]
    global args
    args = args_
    #model loading
    cudnn.benchmark = True
    

    #resume
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        start_epoch = int(args.resume.split('.')[-3].split('_')[-1])+1
        load_pruned(model, args.resume, start_epoch, schedualer, args.load_device)
    elif args.pretrain:
        #start_epoch = int(args.pretrain.split('.')[-3].split('_')[-1])+1
        load_pretrained(model, args.pretrain, rm_layers, args.load_device) #, start_epoch, schedualer
        
    #trainning parameters
    epochs = len(learning_rate)

    
    #loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if teacher_model:
        print("training with teacher network")
        teacher_model.eval()
        criterion2 = nn.KLDivLoss(reduction='batchmean').cuda()
    else:
        criterion2 = None
        
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)


    if not args.evaluate:
        #init tensorboradX
        if args.saveboard:
            if not args.comment:
                args.comment = input("please describe the training: ")
            current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())+args.comment
            train_board_path = os.path.join(args.tensorboard, current_time, 'Train')
            val_board_path = os.path.join(args.tensorboard, current_time, 'Eval')
            os.makedirs(train_board_path)
            os.makedirs(val_board_path)
            train_writer = SummaryWriter(train_board_path)
            eval_writer = SummaryWriter(val_board_path)
        else:
            train_writer = None
            eval_writer = None
        #network result
        checkpoint_path = os.path.join(args.checkpoint_path, current_time)
        os.makedirs(checkpoint_path)
    else:
        args.saveboard = False
        validate(val_loader, model, criterion, start_epoch, None)
        return 

    if schedualer:
        args.save_all_checkpoint = True
        print("save all checkpoints")

    best_acc1 = 0
    #start training
    for epoch in range(start_epoch, epochs+1):

        if schedualer:
            if epoch <= schedualer.get_steps():
                schedualer.schedual()
                learning_rate[epoch-1] = schedualer.get_lr()
                if schedualer.need_forzen:
                    need_forzen = schedualer.get_need_forzen()
                    for name, parm in model.named_parameters():
                        status = True
                        for layer in need_forzen:
                            if layer in name:
                                parm.requires_grad = True
                                status = False
                                print("not forzen:", name)
                                break
                        if status:
                            parm.requires_grad = False  
            else:
                if schedualer.need_forzen:
                    for name, parm in model.named_parameters():
                        parm.requires_grad = True
                    print("not forzen all layers")        

        #forzen, forzen_multi_gpu=0, forzen_epoch =[1]*60 + [2]*30, need_forzen=["conv1", "conv2"]
        if len(forzen_epoch) > 0:
            # print("forzen:", need_forzen[:forzen_epoch[epoch-1]])
            for name, parm in model.named_parameters():
                status = True
                for layer in need_forzen[:forzen_epoch[epoch-1]]:
                    if layer in name:
                        parm.requires_grad = False
                        status = False
                        break
                if status:
                    parm.requires_grad = True
                    print("not forzen:", name)
                # print(name, parm.requires_grad)          

            

        #adjust the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate[epoch-1] #lr #
        print("learning rate:", learning_rate[epoch-1])

        #train one epoch
        train(train_loader, model, criterion, optimizer, epoch, train_writer, teacher_model, criterion2) #, teacher_device

        #evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, eval_writer)

        # remember best acc@1 and save checkpoint
        is_best = prec1 > best_acc1
        best_acc1 = max(prec1, best_acc1)
        if is_best or args.save_all_checkpoint:
            filepath = os.path.join(checkpoint_path, 'epoch_{}_{}.pth.tar'.format(int(prec1), epoch))
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'top1':prec1}, filepath)

        

def train(train_loader, model, criterion, optimizer, epoch, train_writer, teacher_model, criterion2):#, teacher_device
    # batch_time = AverageMeter() # 
    # data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    nums_iters = len(train_loader)
    total_iters = nums_iters*(epoch-1)

    model.train()
    if teacher_model:
        T = 2
        alpha = 0.5 / (epoch//10 + 1)
    end = time.time()
    for it, (inputs, targets) in enumerate(train_loader):
        data_time = time.time()- end
        #prepare data to gpu
        input_var = torch.autograd.Variable(inputs).cuda(non_blocking=True)
        target_var = torch.autograd.Variable(targets).cuda(non_blocking=True)

        #start gpu compute
        output = model(input_var)
        loss = criterion(output, target_var)
        if teacher_model:
            with torch.no_grad():
                output2 = teacher_model(input_var)#to(teacher_device)
            loss2 = criterion2(F.log_softmax(output/T, dim=1), F.softmax(output2/T, dim=1)) * T * T
            loss = loss * (1 - alpha) + loss2 * alpha

        #measure accuracy
        prec = accuracy(output.detach(), target_var.data, top=(1, 5))
        num_batch = target_var.shape[0]
        losses.update(loss.detach().item(), num_batch)
        top1.update(prec[0].item(), num_batch)
        top5.update(prec[1].item(), num_batch)
    
        #compute gradient and backword
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        #print the loss and accuracy
        if it % args.show_freq == 0:
            print('Epoch:[{epoch}][{iter}/{nums_iters}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Time {batch_time:.3f} ({data_time:.3f})\t' 
                  '{local_time}'.format(
                   epoch=epoch, iter=it, nums_iters=nums_iters, batch_time=batch_time, data_time=data_time, 
                   loss=losses, top1=top1, top5=top5, local_time = time.strftime("%H:%M:%S", time.localtime())))
        #add to tensorboardX
        if args.saveboard:
            train_writer.add_scalar("Loss/Loss", losses.avg, total_iters+it)
            train_writer.add_scalar("Accuracy/Top1", top1.avg, total_iters+it)
            train_writer.add_scalar("ZAccuracy/Top5", top5.avg, total_iters+it)
        
    #add to tensorboard
    if args.saveboard:
        train_writer.add_scalar("Loss/Loss_epoch", losses.avg, epoch)
        train_writer.add_scalar("Accuracy/Top1_epoch", top1.avg, epoch)
        train_writer.add_scalar("ZAccuracy/Top5_epoch", top5.avg, epoch)

    
def validate(val_loader, model, criterion, epoch, eval_writer):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    nums_iters = len(val_loader)

    model.eval()


    end = time.time()
    with torch.no_grad():
        for it, (inputs, targets) in enumerate(val_loader):
            input_var = torch.autograd.Variable(inputs).cuda(non_blocking=True)
            target_var = torch.autograd.Variable(targets).cuda(non_blocking=True)

            #compute
            output = model(input_var)
            loss = criterion(output, target_var)

            #measure
            prec = accuracy(output.detach(), target_var.data, top=(1, 5), class_acc = args.class_acc)
            num_batch = target_var.size(0)
            losses.update(loss.detach().item(), num_batch)
            top1.update(prec[0].item(), num_batch)
            top5.update(prec[1].item(), num_batch)

            batch_time = time.time() - end
            end = time.time()
            if it % args.show_freq == 0:
                print('Epoch:[{epoch}][{iter}/{nums_iters}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    'Time {batch_time:.3f}\t'#   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    '{local_time}'.format(
                    epoch=epoch, iter=it, nums_iters=nums_iters, batch_time=batch_time, #data_time=data_time, 
                    loss=losses, top1=top1, top5=top5, local_time = time.strftime("%H:%M:%S", time.localtime())))
                # print(args.class_acc[:10])
        print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    #add to tensorboard
    if args.saveboard:
        eval_writer.add_scalar("Loss/Loss_epoch", losses.avg, epoch)
        eval_writer.add_scalar("Accuracy/Top1_epoch", top1.avg, epoch)
        eval_writer.add_scalar("ZAccuracy/Top5_epoch", top5.avg, epoch)

    

    return top1.avg

if __name__ == "__main__":
    pass



