# -*- coding: utf-8 -*-
'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging

from models import *
from utils import *
from data_loader import *
import itertools
import pdb
from plot_results import *


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
fmt_str = '%(levelname)s - %(message)s'
fmt_file = '[%(levelname)s]: %(message)s'
# best_acc = 0

def parse_arguments():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--cloud_lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--cloud_epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--net', type=str, default='res8')
    parser.add_argument('--num_workers', default=2, type=int, help='number of edge workers')
    parser.add_argument('--num_rounds', default=1, type=int, help='number of rounds')
    parser.add_argument('--cloud', type=str, default='res18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--cloud_batch_size', type=int, default='128')
    parser.add_argument('--workspace', default='')
    parser.add_argument("--split", default=0.1, type=float, help="split training data")
    parser.add_argument("--split_classes", action='store_true', help='split the number of classes to reduce difficulty')
    parser.add_argument("--baseline", action='store_true', help='Perform only training for collection baseline')
    parser.add_argument('--temperature', default=1, type=float, help='temperature for distillation')
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--two', action='store_true')
    parser.add_argument('--iid', action='store_true', help="test iid case")
    parser.add_argument('--alternate', action='store_true', help="test the alternative case")
    parser.add_argument('--public_distill', action='store_true', help="use public data to distill")
    parser.add_argument('--exist_loader', action='store_true', help="there is exist loader")
    parser.add_argument('--save_loader', action='store_true', help="save trainloaders")
    parser.add_argument('--alpha', default=100, type=float, help='alpha for iid setting')
    parser.add_argument('--lamb', default=0.5, type=float, help='lambda for distillation')
    parser.add_argument('--public_percent', default=0.5, type=float, help='percentage training data to be public')
    parser.add_argument('--distill_percent', default=1, type=float, help='percentage of public data use for distillation')
    parser.add_argument('--selection', action='store_true', help="enable selection method")
    parser.add_argument('--use_pseudo_labels', action='store_true', help="enable selection method")
    parser.add_argument('--add_cifar10', action='store_true', help="add cifar10 to distillation")
    parser.add_argument('--finetune', action='store_true', help="finetune the cloud model")
    parser.add_argument('--finetune_epoch', default=10, type=int, help='number of epochs for finetune')
    parser.add_argument('--finetune_percent', default=0.2, type=float, help='percentage of data to finetune')


    args = parser.parse_args()

    return args

def defrost_net(net):
    for param in net.parameters():
        param.requires_grad = True
    net.train()
    return net

def freeze_net(net):
    # freeze the layers of a model
    for param in net.parameters():
        param.requires_grad = False
    # set the teacher net into evaluation mode
    net.eval()
    return net

def freeze_except_last_layer(net):
    # freeze the layers of a model
    for param in net.parameters():
        param.requires_grad = False
    
    # Resnet18 uses linear instead of fc layer as the last layer
    # for param in net.fc.parameters():
    #     param.requires_grad = True

    for param in net.linear.parameters():
        param.requires_grad = True    

    for name, param in net.named_parameters():
        print(name, param.requires_grad)

    return net

def print_layers(net):
    for name, param in net.named_parameters():
        print(name, param.requires_grad)
    
# Training
def train(epoch, round, net, criterion, optimizer, trainloader, device):
    logger.debug('\nEpoch: %d' % epoch)
    net.train()
    # print_layers(net)
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += torch.sum(predicted == targets).float()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss/(batch_idx+1)

def test(epoch, args, net, criterion, testloader, device, msg, save_checkpoint=True):

    best_acc = 0
    # global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            # logger.debug(f"outputs.max(1): {outputs.max(1)}\n")
            logger.debug(f"predicted: {predicted}\n")
            logger.debug(f"targets: {targets}\n")
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # correct += torch.sum(predicted == targets.data).float()            
            logger.debug(f"correct={correct} / total {total}\n")
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    #TODO: why not save checkpoint every experiment? 
    acc = 100.*correct/total
    logger.debug(f"acc: {acc}, best_acc: {best_acc}")

    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if save_checkpoint:
            logger.debug('Saving..')
            checkpoint_dir = 'results/' + args.workspace + '/checkpoint/'
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            torch.save(state, checkpoint_dir + msg +'_ckpt.t7')
        best_acc = acc
            
    return acc, best_acc
        

def build_model_from_name(name, num_classes, device):

    print(name, type(name))
    print(name == 'res8')

    if name == 'res6':
        net = resnet6(num_classes=num_classes)

    elif name == 'res8':
        net = resnet8(num_classes=num_classes)    

    elif name == 'res50':
        net = resnet50(num_classes=num_classes)

    elif name == 'res20':
        net = resnet20(num_classes=num_classes)

    elif name == 'res18':
        net = resnet18(num_classes=num_classes)
    
    elif name == 'lenet':
        net = LeNet()

    elif name =='vgg':
        net = VGG('VGG19')

    else:
        print('Not supported model')
        exit()

    print_total_params(net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

    return net

def save_figure(path, csv_name):

    # pdb.set_trace()
    data_to_plot = {}
    data_to_plot['data'] = collect_data(path + '/' + csv_name)
    print(data_to_plot)
    plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Accuracy', output= path + '/' + 'result.png')


def check_model_trainable(nets):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = True

        net.train()
        
    return nets

def run_train(net, round, args, trainloader, testloader, worker_num, device, msg):
    
    list_loss = []
    strtime = get_time()
    criterion_edge = nn.CrossEntropyLoss()
    optimizer_edge = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    csv_name = 'acc_'  + 'worker_' + str(worker_num) + '_' + strtime + '.csv'
    path = 'results/' + args.workspace
    
    # end_epoch = args.finetune_epoch if args.finetune else args.epoch
    # print(end_epoch)
    for epoch in range(start_epoch, start_epoch + args.epoch):
    # for epoch in range(start_epoch, start_epoch + end_epoch):
        trainloss = train(epoch, round, net, criterion_edge, optimizer_edge, trainloader, device)
        acc, best_acc = test(epoch, args, net, criterion_edge, testloader, device, msg)
        logger.debug(f"The result is: {acc}")
        # write_csv('acc_' + args.workspace + '_worker_' + str(worker_num) + '_' + strtime + '.csv', str(acc))
        write_csv(path, csv_name, str(acc))
        list_loss.append(trainloss)
    # pdb.set_trace()
    # save_figure(path, csv_name)
    
    logger.debug("===> BEST ACC. PERFORMANCE: %.2f%%" % (best_acc))

def run_finetune(net, round, args, trainloader, testloader, device, msg):
    
    """
    This is a copy of the run_train function, only changing the args.epoch to args.finetune_epoch
    It is quite redundant, but I don't want to mess up with the training epochs
    """
    
    list_loss = []
    criterion_edge = nn.CrossEntropyLoss()
    optimizer_edge = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    csv_name = 'cloud_finetune.csv'
    path = 'results/' + args.workspace

    for epoch in range(start_epoch, start_epoch + args.finetune_epoch):
        trainloss = train(epoch, round, net, criterion_edge, optimizer_edge, trainloader, device)
        acc, best_acc = test(epoch, args, net, criterion_edge, testloader, device, msg)
        logger.debug(f"The result is: {acc}")
        write_csv(path, csv_name, str(acc))
        list_loss.append(trainloss)

    # pdb.set_trace()
    # save_figure(path, csv_name)
    
    logger.debug("===> BEST ACC. PERFORMANCE: %.2f%%" % (best_acc))


def run_train_non_iid(net, round, args, trainloader, testloader, testloader_local, worker_num, device, msg):
    """ Add this the perform extra step for local classes accuracy"""
    
    list_loss = []
    strtime = get_time()
    criterion_edge = nn.CrossEntropyLoss()
    optimizer_edge = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    csv_name = 'acc_'  + 'worker_' + str(worker_num) + '_' + strtime + '.csv'
    csv_name_local = 'acc_'  + 'worker_' + str(worker_num) + '_' + strtime + '_' + 'local' + '.csv'

    path = 'results/' + args.workspace
        
    for epoch in range(start_epoch, start_epoch + args.epoch):        
        trainloss = train(epoch, round, net, criterion_edge, optimizer_edge, trainloader, device)
        acc, best_acc = test(epoch, args, net, criterion_edge, testloader, device, msg)
        logger.debug(f"The result is: {acc}")
        write_csv(path, csv_name, str(acc))

        acc_local, _ = test(epoch, args, net, criterion_edge, testloader_local, device, msg)
        write_csv(path, csv_name_local, str(acc_local))

        list_loss.append(trainloss)
    # pdb.set_trace()
    # save_figure(path, csv_name)
    
    logger.debug("===> BEST ACC. PERFORMANCE: %.2f%%" % (best_acc))



def distill(epoch, edge_net, cloud_net, optimizer, trainloader, device, lambda_=1):
    
    logger.debug(f"\nEpoch: {epoch}, Lambda: {lambda_}")
    cloud_net.train()
    logger.debug(f"model.train={cloud_net.training}")
    train_loss = 0
    correct = 0
    total = 0
    loss_fun = nn.CrossEntropyLoss()
    kd_fun = nn.KLDivLoss(reduction='sum')
    T = 1

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        out_s = cloud_net(inputs)
        out_t = edge_net(inputs)
        batch_size = out_s.shape[0]

        # Check the parameters before and after training step
        Before = list(cloud_net.parameters())[0].clone()

        edge_before = list(edge_net.parameters())[0].clone()

        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)

        loss_kd = kd_fun(s_max, t_max) / batch_size
        loss = loss_fun(out_s, targets)

        loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

        loss_kd.backward()
        optimizer.step()

        After = list(cloud_net.parameters())[0].clone()
        edge_after = list(edge_net.parameters())[0].clone()

        logger.debug(f'Cloud: {torch.equal(Before.data, After.data)}')
        logger.debug(f'Edge:  {torch.equal(edge_before.data, edge_after.data)}')

        train_loss += loss_kd.item()
        logger.debug(f"loss_kd.item() {loss_kd.item()}\n")

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f ' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)


def count_targets(targets):
    alpha, beta, gamma = 0, 0, 0
    batch_size = targets.shape[0]
    # Now only works for 30 classes
    for target in targets:
        if 0 <= target.item() <= 9:
            alpha += 1
        elif 10 <= target.item() <= 19:
            beta += 1
        else:
            gamma += 1
    
    alpha /= batch_size
    beta /= batch_size
    gamma /= batch_size

    logger.debug(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}")
    return alpha, beta, gamma 

def distill_from_concat_multi_workers(
    epoch, 
    edge_net_0, 
    edge_net_1, 
    edge_net_2, 
    cloud_net, 
    optimizer, 
    trainloader_concat, 
    device,
    average_method,
    selection=False,
    use_pseudo_labels=False,
    lambda_=0.5,
    )->float:
    """ 
    This function is a hack for using 30 classes training data for distillation
    Here I directly use the trainloader that contains the first 30 classes, so it can only provide a baseline
    """
    
    logger.debug('\nEpoch: %d' % epoch)
    cloud_net.train()
    logger.debug(f"model.train={cloud_net.training}")
    train_loss = 0
    correct = 0
    total = 0
    loss_fun = nn.CrossEntropyLoss()
    kd_fun = nn.KLDivLoss(reduction='sum')
    T = 1

    counter = 0
    logger.debug(f"Lambda: {lambda_}")
    for batch_idx, (inputs, targets) in enumerate(trainloader_concat):
        inputs, targets = inputs.to(device), targets.to(device)
        alpha, beta, gamma = count_targets(targets)
        optimizer.zero_grad()
        counter += 1
        logger.debug(f"Counter: {counter}")
        out_s = cloud_net(inputs)
        # Use the knowledge from the correct edge model
        # Only check the first sample is enough
        # if 0 <= targets[0].item() <= 9:
        #     logger.debug(f"Batch idx: {batch_idx}, Worker 0 data")
        #     out_t = edge_net_0(inputs)
        # elif 10 <= targets[0].item() <= 19:
        #     logger.debug(f"Batch idx: {batch_idx}, Worker 1 data")
        #     out_t = edge_net_1(inputs)
        # elif 20 <= targets[0].item() <= 29:
        #     logger.debug(f"Batch idx: {batch_idx}, Worker 2 data")
        #     out_t = edge_net_2(inputs)

        batch_size = out_s.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)

        if selection: # direct the data to edge worker based on classes 
            out_t_temp = []
            logger.debug(f"Selection")
            logger.debug(f"inputs: {inputs.shape}, targets: {targets.shape}")
            for input, target in zip(inputs, targets):
                # need to use unsqueeze to change a 3-dimensional image to 4-dimensional
                # [3,32,32] -> [1,3,32,32]
                input = input.unsqueeze(0)
                # pdb.set_trace()
                if 0 <= target.item() <= 9 :
                    out_t_temp.append (edge_net_0(input))
                elif 10 <= target.item() <= 19 :
                    out_t_temp.append (edge_net_1(input))
                else:
                    out_t_temp.append (edge_net_2(input))
                # out_t_temp.append (edge_net_0(input))
            logger.debug(f"len out_t_temp: {len(out_t_temp)}")
            out_t = torch.cat(out_t_temp, dim=0) # use dim 0 to stack the tensors
            assert out_t.shape[1] == 100, f"the shape is {out_t.shape}, should be (x, 100)"
            t_max = F.softmax(out_t / T, dim=1)

            if use_pseudo_labels:
                _, pseudo_labels = out_t.max(1)


        #TODO: use an array of edge models instead later
        # you can use torch.zeros(128,100) to create the placeholder 
        else:
            out_t_0 = edge_net_0(inputs)
            t_max_0 = F.softmax(out_t_0 / T, dim=1)

            out_t_1 = edge_net_1(inputs)
            t_max_1 = F.softmax(out_t_1 / T, dim=1)

            out_t_2 = edge_net_2(inputs)
            t_max_2 = F.softmax(out_t_2 / T, dim=1)

            if average_method == 'equal':
                t_max = (t_max_0 + t_max_1 + t_max_2) / 3
            else: # weighted
                logger.debug(f"Weighted average")
                t_max = alpha * t_max_0 + beta * t_max_1 + gamma * t_max_2

        # logger.debug(f"s_max: {s_max}")    
        # logger.debug(f"t_max: {t_max}")

        loss_kd = kd_fun(s_max, t_max) / batch_size
        loss = loss_fun(out_s, targets)
        
        if use_pseudo_labels:
            logger.debug(f"Enable pseudo labels")
            loss_sd = loss_fun(out_s, pseudo_labels)
            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd + 0.5 * loss_sd

        else:
            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

        loss_kd.backward()
        optimizer.step()

        train_loss += loss_kd.item()
        
    return train_loss/(batch_idx+1)

def distill_from_concat_two_workers(
    epoch, 
    edge_net_0, 
    edge_net_1, 
    cloud_net, 
    optimizer, 
    trainloader_concat, 
    device,
    split,
    average_method,
    selection=False,
    use_pseudo_labels=False,
    lambda_=1,
    )->float:
    """ 
    This function is a hack for using 30 classes training data for distillation
    Here I directly use the trainloader that contains the first 30 classes, so it can only provide a baseline
    """
    
    logger.debug('\nEpoch: %d' % epoch)
    cloud_net.train()
    logger.debug(f"model.train={cloud_net.training}")
    train_loss = 0
    correct = 0
    total = 0
    loss_fun = nn.CrossEntropyLoss()
    kd_fun = nn.KLDivLoss(reduction='sum')
    T = 1

    # consider cifar100 case for now
    total_classes = 100
    num_workers = 2
    worker_ids = [x for x in range(num_workers)]
    num_classes = int(split * total_classes)
    bounds = []
    for worker_id in worker_ids:
        start = worker_id * num_classes + 0
        end = worker_id * num_classes + num_classes - 1
        bounds.append([start, end])

    counter = 0
    logger.debug(f"Lambda: {lambda_}")
    for batch_idx, (inputs, targets) in enumerate(trainloader_concat):
        inputs, targets = inputs.to(device), targets.to(device)
        alpha, beta, gamma = count_targets(targets)
        optimizer.zero_grad()
        counter += 1
        logger.debug(f"Counter: {counter}")
        out_s = cloud_net(inputs)

        batch_size = out_s.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)

        if selection: # direct the data to edge worker based on classes 
            out_t_temp = []
            logger.debug(f"Selection")
            logger.debug(f"inputs: {inputs.shape}, targets: {targets.shape}")
            # pdb.set_trace()
            for input, target in zip(inputs, targets):
                # need to use unsqueeze to change a 3-dimensional image to 4-dimensional
                # [3,32,32] -> [1,3,32,32]
                input = input.unsqueeze(0)
                
                # bounds is the list of the lower and upper bound of each worker
                # e.g., [[0, 1], [2, 3]]

                if bounds[0][0] <= target.item() <= bounds[0][1] :
                    # logger.debug("first worker data")
                    out_t_temp.append (edge_net_0(input))
                elif bounds[1][0] <= target.item() <= bounds[1][1] :
                    # logger.debug("second worker data")
                    out_t_temp.append (edge_net_1(input))
                else:
                    logger.debug("Should not happen")
                    exit()

                # out_t_temp.append (edge_net_0(input))
            logger.debug(f"len out_t_temp: {len(out_t_temp)}")
            out_t = torch.cat(out_t_temp, dim=0) # use dim 0 to stack the tensors
            assert out_t.shape[1] == 100, f"the shape is {out_t.shape}, should be (x, 100)"
            t_max = F.softmax(out_t / T, dim=1)

            if use_pseudo_labels:
                _, pseudo_labels = out_t.max(1)


        #TODO: use an array of edge models instead later
        # you can use torch.zeros(128,100) to create the placeholder 
        else:
            out_t_0 = edge_net_0(inputs)
            t_max_0 = F.softmax(out_t_0 / T, dim=1)

            out_t_1 = edge_net_1(inputs)
            t_max_1 = F.softmax(out_t_1 / T, dim=1)

            if average_method == 'equal':
                logger.debug("For iid case")
                t_max = (t_max_0 + t_max_1 ) / 2
            else: # weighted
                logger.debug(f"Weighted average")
                t_max = alpha * t_max_0 + beta * t_max_1

        # logger.debug(f"s_max: {s_max}")    
        # logger.debug(f"t_max: {t_max}")

        loss_kd = kd_fun(s_max, t_max) / batch_size
        loss = loss_fun(out_s, targets)
        
        if use_pseudo_labels:
            logger.debug(f"Enable pseudo labels")
            loss_sd = loss_fun(out_s, pseudo_labels)
            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd + 0.5 * loss_sd

        else:
            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

        loss_kd.backward()
        optimizer.step()

        train_loss += loss_kd.item()
        
    return train_loss/(batch_idx+1)


def distill_from_multi_workers(
    epoch,  
    edge_nets, 
    cloud_net, 
    optimizer, 
    trainloader_concat, 
    device,
    num_workers,
    split,
    distill_percent, 
    average_method,
    select_mode,
    selection=False,
    use_pseudo_labels=False,
    lambda_=1,
    )->float: 
    
    logger.debug('\nEpoch: %d' % epoch)
    cloud_net.train()
    # logger.debug(f"model.train={cloud_net.training}")
    train_loss = 0
    correct = 0
    total = 0
    loss_fun = nn.CrossEntropyLoss()
    kd_fun = nn.KLDivLoss(reduction='sum')
    T = 1

    # consider cifar100 only for now
    total_classes = 100
    worker_ids = [x for x in range(num_workers)]
    num_classes = int(split * total_classes)
    bounds = []
    for worker_id in worker_ids:
        start = worker_id * num_classes + 0
        end = worker_id * num_classes + num_classes - 1
        bounds.append([start, end])

    print(bounds)

    counter = 0
    logger.debug(f"Lambda: {lambda_}")
    for batch_idx, (inputs, targets) in enumerate(trainloader_concat):
        inputs, targets = inputs.to(device), targets.to(device)
        # logger.debug(f"device = {device}")
        alpha, beta, gamma = count_targets(targets)
        optimizer.zero_grad()
        counter += 1
        logger.debug(f"Counter: {counter}")
        out_s = cloud_net(inputs)

        batch_size = out_s.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)

        if selection: # direct the data to edge worker based on classes 
            out_t_temp = []
            logger.debug(f"Selection")
            logger.debug(f"inputs: {inputs.shape}, targets: {targets.shape}")
            # pdb.set_trace()
            for input, target in zip(inputs, targets):
                # need to use unsqueeze to change a 3-dimensional image to 4-dimensional
                # [3,32,32] -> [1,3,32,32]
                input = input.unsqueeze(0)
                if select_mode == "guided": 
                    # bounds is the list of the lower and upper bound of each worker
                    # e.g., [[0, 1], [2, 3]]
                    for i, bound in enumerate(bounds):
                        # print(f"bound {bound}, target {target.item()}")
                        if int(target.item()) in bound:
                            out_t_temp.append (edge_nets[i](input))
                        else:
                            continue

                elif select_mode == "similarity":
                    pass
                    
                else:
                    logger.debug("No select model provided")
                    exit()

            logger.debug(f"len out_t_temp: {len(out_t_temp)}")
            out_t = torch.cat(out_t_temp, dim=0) # use dim 0 to stack the tensors
            assert out_t.shape[1] == 100, f"the shape is {out_t.shape}, should be (x, 100)"
            t_max = F.softmax(out_t / T, dim=1)

            if use_pseudo_labels:
                _, pseudo_labels = out_t.max(1)

        #TODO: use an array of edge models instead later
        # you can use torch.zeros(128,100) to create the placeholder 
        else:

            out_t = torch.zeros((batch_size, total_classes), device=device)
            t_max = torch.zeros((batch_size, total_classes), device=device)

            for edge_net in edge_nets:
                edge_net = edge_net.to(device)
                out_t += edge_net(inputs)
                t_max += F.softmax(out_t / T, dim=1)

            if average_method == 'equal':
                logger.debug("Equal weights")
                # t_max = (t_max_0 + t_max_1 ) / 2
                t_max = t_max / num_workers
            else: # weighted, performance is not good, not used
                logger.debug("Not support weighted average now")
                # logger.debug(f"Weighted average")
                # t_max = alpha * t_max_0 + beta * t_max_1

        # logger.debug(f"s_max: {s_max}")
        # logger.debug(f"t_max: {t_max}")

        loss_kd = kd_fun(s_max, t_max) / batch_size
        loss = loss_fun(out_s, targets)
        
        if use_pseudo_labels:
            logger.debug(f"Enable pseudo labels")
            loss_sd = loss_fun(out_s, pseudo_labels)
            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd + 0.5 * loss_sd

        else:
            # loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd
            # mix the batch of labeled and unlabeled data (11/16)
            if batch_idx <= len(trainloader_concat) * distill_percent:
                lambda_ = 1
            else:
                lambda_ = 0.75

            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

        loss_kd.backward()
        optimizer.step()

        train_loss += loss_kd.item()
        
    return train_loss/(batch_idx+1)


def run_concat_distill_two(
    edge_0, 
    edge_1, 
    cloud, 
    args, 
    trainloader_cloud, 
    testloader_cloud,
    worker_num, 
    device
    )->nn.Module:
    """Concatenate the three datase from the edge workers together and test distillation with lambda set to 0"""
    
    logger.debug("From concat distill")
    strtime = get_time()
    list_loss = []

    frozen_edge_0 = freeze_net(edge_0)
    frozen_edge_0 = frozen_edge_0.to(device)
    
    frozen_edge_1 = freeze_net(edge_1)
    frozen_edge_1 = frozen_edge_1.to(device)

    criterion_cloud = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(cloud.parameters(), lr=args.cloud_lr, momentum=0.9, weight_decay=5e-4)
    else:
        logger.debug("Using adam optimizer")
        optimizer = optim.Adam(cloud.parameters(), lr=args.cloud_lr, weight_decay=5e-4)

    for epoch in range(args.cloud_epoch):
        
        # TODO: need to change the distill from multi workers to support array of trainloaders and workers. Now it is a hacky method 
        # TODO: merge selection into average method
        trainloss = distill_from_concat_two_workers(epoch, 
                                                    frozen_edge_0, 
                                                    frozen_edge_1, 
                                                    cloud, 
                                                    optimizer, 
                                                    trainloader_cloud, 
                                                    device, 
                                                    args.split,
                                                    average_method='equal', 
                                                    selection=args.selection, 
                                                    use_pseudo_labels=args.use_pseudo_labels, 
                                                    lambda_=args.lamb)
        
        acc, best_acc = test(epoch, args, cloud, criterion_cloud, testloader_cloud, device, 'cloud')

        logger.debug(f"The result is: {acc}")
        write_csv('results/' + args.workspace, 'distill_concat_' + strtime + '.csv', str(acc))
        list_loss.append(trainloss)

    logger.debug("===> BEST ACC. DISTILLATION: %.2f%%" % (best_acc))

    return cloud


def run_concat_distill_multi(
    edge_nets,  
    cloud, 
    args, 
    trainloader_cloud, 
    testloader_cloud,
    worker_num, 
    device,
    prefix
    )->nn.Module:
    """Concatenate the three datase from the edge workers together and test distillation with lambda set to 0"""
    
    logger.debug("From concat distill")
    strtime = get_time()
    list_loss = []

    frozen_edge_nets = []
    for edge_net in edge_nets:
        frozen_edge_net = freeze_net(edge_net)
        frozen_edge_nets.append(frozen_edge_net)

    criterion_cloud = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(cloud.parameters(), lr=args.cloud_lr, momentum=0.9, weight_decay=5e-4)
    else:
        logger.debug("Using adam optimizer")
        optimizer = optim.Adam(cloud.parameters(), lr=args.cloud_lr, weight_decay=5e-4)

    for epoch in range(args.cloud_epoch):
        
        # TODO: need to change the distill from multi workers to support array of trainloaders and workers. Now it is a hacky method 
        # TODO: merge selection into average method
        trainloss = distill_from_multi_workers(epoch, 
                                                frozen_edge_nets,
                                                cloud, 
                                                optimizer, 
                                                trainloader_cloud, 
                                                device, 
                                                args.num_workers,
                                                args.split,
                                                args.distill_percent,
                                                average_method='equal', 
                                                select_mode='guided',
                                                selection=args.selection, 
                                                use_pseudo_labels=args.use_pseudo_labels, 
                                                lambda_=args.lamb)

        acc, best_acc = test(epoch, args, cloud, criterion_cloud, testloader_cloud, device, 'cloud')

        logger.debug(f"The result is: {acc}")
        # write_csv('results/' + args.workspace, 'distill_concat_' + strtime + '.csv', str(acc))
        write_csv('results/' + args.workspace, prefix + strtime + '.csv', str(acc))

        list_loss.append(trainloss)

    logger.debug("===> BEST ACC. DISTILLATION: %.2f%%" % (best_acc))

    return cloud


def run_distill(
    edge, 
    cloud, 
    args, 
    trainloader, 
    testloader, 
    worker_num, 
    device
    )->nn.Module:
    """Wrapper function to distill a network."""

    strtime = get_time()
    list_loss = []

    frozen_edge = freeze_net(edge)
    frozen_edge = frozen_edge.to(device)
    criterion_cloud = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(cloud.parameters(), lr=args.cloud_lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(cloud.parameters(), lr=args.cloud_lr, weight_decay=5e-4)

    for epoch in range(args.epoch):
        
        trainloss = distill(epoch, frozen_edge, cloud, optimizer, trainloader, device, lambda_=args.lamb)
        acc, best_acc = test(epoch, args, cloud, criterion_cloud, testloader, device, 'cloud')

        logger.debug(f"The result is: {acc}")
        # write_csv('distill_acc_' + args.workspace + '_' + 'worker_' + str(worker_num) + '_' + strtime + '.csv', str(acc))
        write_csv('results/' + args.workspace, 'distill_' + str(worker_num) + '_' + strtime + '.csv', str(acc))

        list_loss.append(trainloss)

    logger.debug(f"Linear.weight after distillation from worker {worker_num}: {cloud.linear.weight.data}")
    logger.debug("===> BEST ACC. DISTILLATION: %.2f%%" % (best_acc))

    return cloud

def test_only(
    net, 
    testloader,
    device
    )->None:
    """Test a model."""
    
    strtime = get_time()
    criterion = nn.CrossEntropyLoss()
    acc, best_acc = test(0, net, criterion, testloader, device, 'test')
    logger.debug(f"The result is: {acc}")
    # write_csv('acc_' + args.workspace +  '_test_other_ten' + strtime + '.csv', str(acc))
    write_csv('results/' + args.workspace, 'acc_' +  'test_other_ten' + strtime + '.csv', str(acc))
    logger.debug("===> BEST ACC. ON OTHER TEN CLASSES: %.2f%%" % (best_acc))

    
if __name__ == "__main__":

    args = parse_arguments()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    torch.manual_seed(0)
    # print(args)

    root = 'results/' + args.workspace
    check_dir(root)
    c_handler, f_handler = get_logger_handler(root + '/output.log')
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info(args)
    
    # Data
    logger.debug('==> Preparing data..')
    num_classes = 10 if args.dataset == 'cifar10' else 100

    if args.dataset == 'cifar10':
        trainloader, testloader = get_cifar10_loader(args)

    else:
        logger.info('==> CIFAR-100')
        trainset, testset = get_cifar100()
        transform_test = get_cifar100_transfromtest()

        if args.public_distill: 
            trainset_public, trainset_private = split_train_data(trainset, args.public_percent)

            # Use 10% of the public dataset
            if args.distill_percent != 1:
                
                # do notthing for now (moved insided the distillation function)
                logger.info(f"The distill percent is {args.distill_percent}")
                # use partial public data to distill
                # trainset_public, _ = split_train_data(trainset_public, args.distill_percent)
                # use partial private data to distill
                # trainset_public, _ = split_train_data(trainset_private, args.distill_percent)

        if args.split != 0 and not args.split_classes:
            logger.info(f"Using {int(args.split * 100)}% of training data, classifying all classes")
            trainset_a, trainset_b = split_train_data(trainset, args.split)
            trainloader = torch.utils.data.DataLoader(trainset_a, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        elif args.split != 0 and args.split_classes and not args.baseline:
            # Use all data
            logger.info(f"is it here Using {int(args.split * 100)}% of training data and classifying {int(args.split * 100)}%")

            if args.exist_loader:
                trainloader = torch.load('trainloader_first_10cls.pth')
                testloader = torch.load('testloader_first_10cls.pth')
            
            else:
                client_classes = int(num_classes * args.split)
                if args.public_distill and not args.iid:

                    # trainloader, testloader = get_worker_data(trainset_private, args, workerid=0)
                    trainloaders = get_subclasses_loaders(trainset_private, args.num_workers, client_classes, num_workers=4, seed=100)
                    # trainloader_all = get_subclasses_loaders(trainset, n_clients=1, client_classes=int(args.num_workers * args.client_classes), num_workers=4, seed=100)
                    testloader_non_iid = get_subclasses_loaders(testset, n_clients=1, client_classes=args.num_workers * client_classes, num_workers=4, seed=100)
                    # logger.debug(testloader_non_iid[0])
                    testloaders = get_subclasses_loaders(testset, args.num_workers, client_classes, num_workers=4, seed=100)
                    trainloader_all = get_subclasses_loaders(trainset_public, n_clients=1, client_classes=args.num_workers * client_classes, num_workers=4, seed=100)

                elif args.public_distill and args.iid:
                    logger.info(f"Using {int(args.split * 100)}% for iid")
                    extract_trainset = extract_classes(trainset_private, args.split, workerid=0)
                    # use 1 thread worker instead of 4 in the single gpu case
                    trainloaders = get_dirichlet_loaders(extract_trainset, n_clients=args.num_workers, alpha=args.alpha, num_workers=1, seed=100)
                    _, testloader_iid = get_worker_data_hardcode(trainset, args.split, workerid=0)

                else:
                    # trainloader, testloader = get_worker_data(trainset, args, workerid=0)
                    trainloaders = get_subclasses_loaders(trainset, args.num_workers, client_classes, num_workers=4, seed=100)
                    trainloader_all = get_subclasses_loaders(trainset, n_clients=1, client_classes=int(args.num_workers * client_classes), num_workers=4, seed=100)

                    # _, testloader_non_iid = get_worker_data_hardcode(trainset, args.split, workerid=0)
                    testloader_non_iid = get_subclasses_loaders(testset, n_clients=1, client_classes=int(args.num_workers * client_classes), num_workers=4, seed=100)

                    # logger.debug(testloader_non_iid[0])
                    testloaders = get_subclasses_loaders(testset, args.num_workers, client_classes, num_workers=4, seed=100)

            if args.save_loader:
                torch.save(trainloader, 'trainloader_first_10cls.pth')
                torch.save(testloader, 'testloader_first_10cls.pth')

            # two workers
            if args.two:    
                logger.info(f"Creating a new loader Using {int(args.split * 100)}% of training data and classifying {int(args.split * 100)}%")               
                
                # get worker data is slow, maybe we can get save the data first. 
                # workerid=1
                if args.exist_loader:
                    trainloader_1 = torch.load('trainloader_second_10cls.pth')
                    testloader_1 = torch.load('testloader_second_10cls.pth')
                    trainloader_2 = torch.load('trainloader_third_10cls.pth')
                    testloader_2 = torch.load('testloader_third_10cls.pth')
                    testloader_20cls = torch.load('testloader_20cls.pth')
                    testloader_30cls = torch.load('testloader_30cls.pth')
                    trainloader_30cls = torch.load('trainloader_30cls.pth')

                    logger.debug("Done loading loaders")

                else:
                    if args.public_distill:
                        logger.debug("In the public_distill condition")
                        # trainloader_private, _ = get_worker_data_hardcode(trainset_private, args.num_workers * args.split, workerid=0)
                        if args.iid:
                            trainloader_public, testloader_public = get_worker_data_hardcode(trainset_public, args.split, workerid=0)
                        
                        else:
                            trainloader_public, testloader_public = get_worker_data_hardcode(trainset_public, args.num_workers * args.split, workerid=0)


                    else:
                        logger.debug("In the else condition")
                        
                if args.save_loader:
                    torch.save(trainloader_1, 'trainloader_second_10cls.pth')
                    torch.save(testloader_1, 'testloader_second_10cls.pth')

                    torch.save(trainloader_2, 'trainloader_third_10cls.pth')
                    torch.save(testloader_2, 'testloader_third_10cls.pth')
                    
                    torch.save(trainloader_30cls, 'trainloader_30cls.pth')
                    torch.save(testloader_20cls_disjoint, 'testloader_20cls_disjoint.pth')
                    torch.save(testloader_20cls, 'testloader_20cls.pth')
                    torch.save(testloader_30cls, 'testloader_30cls.pth')
                    logger.debug("Done saving loaders")
                    exit()

        # use to collect baseline results
        # TODO: simplify the test cases. The test cases are overlapped. 
        # The iid case will never got to run if I don't comment out the large chunk of code above
        
        #################################### For baseline test ####################################
        elif args.split != 0 and args.baseline:
            if not args.iid:
                logger.info(f"Using {int(args.split * 100)}% for simple baseline")
                trainloader, testloader = get_worker_data(trainset, args, workerid=0)

            else: # iid case
                logger.info(f"Using {int(args.split * 100)}% for iid baseline")
                extract_trainset = extract_classes(trainset, args.split, workerid=0)
                        
                # use 1 thread worker instead of 4 in the single gpu case
                trainloaders = get_dirichlet_loaders(extract_trainset, n_clients=args.num_workers, alpha=args.alpha, num_workers=1, seed=100)
                # _, testloader_30cls = get_worker_data_hardcode(trainset, 0.3, workerid=0)
                _, testloader_iid = get_worker_data_hardcode(trainset, args.split, workerid=0)
                # pdb.set_trace()

        ###########################################################################################
        
        elif args.split == 0:
            logger.info(f"Using full training data")
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

        else:
            logger.info('Please check the data split options')
            exit()

    # Model
    logger.debug('==> Building model..')

    nets = []
    for i in range(args.num_workers):
        net = build_model_from_name(args.net, num_classes, device)
        nets.append(net)

    cloud_net = build_model_from_name(args.cloud, num_classes, device)

    if not args.alternate: # if not alternate, then perform sequential distillation
        if args.resume:
            nets[0] = load_edge_checkpoint_fullpath(nets[0], 'results/public_percent_0.5_2_cls_adam_lambda_1/checkpoint/edge_0_ckpt.t7')
        run_distill(nets[0], cloud_net, args, trainloader, testloader, worker_num=0, device=device)
        exit()

    if args.two:
        if args.resume:
            # net_1 = load_edge_checkpoint_fullpath(net_1, 'results/public_percent_0.5_2_cls_adam_lambda_1/checkpoint/edge_1_ckpt.t7')
            # net_2 = load_edge_checkpoint(net_2, 'res8_edge_2_ckpt.t7')
            nets[1] = load_edge_checkpoint_fullpath(nets[1], 'results/public_percent_0.5_2_cls_adam_lambda_1/checkpoint/edge_1_ckpt.t7')
        else:
            if args.iid:
                """
                commented out the unnecessary trainings. It's time to merge iterative and seperate together?
                Starting from the iid case using public data to distill 
                """
                # run_train(nets[1], args, trainloaders[1], testloader_iid, 1, device, 'edge_1')

                for round in range(args.num_rounds):
                    
                    logger.debug(f"############# round {round} #############")
                    nets = check_model_trainable(nets)
                    for i in range(0, args.num_workers, 1):
                        run_train(nets[i], round, args, trainloaders[i], testloader_iid, i, device, 'edge_' + str(i))
                    
                    logger.debug("Distilling with public data")
                    run_concat_distill_multi(nets, cloud_net, args, trainloader_public, testloader_public, worker_num=0, device=device, prefix='distill_')

                    if args.finetune:
                        # cloud_net = freeze_except_last_layer(cloud_net)
                        trainset_finetune, _ = split_train_data(trainset_public, args.finetune_percent)
                        finetune_loader = get_loader(trainset_finetune, args)
                        
                        run_finetune(cloud_net, 0, args, finetune_loader, testloader_public, device, 'cloud_finetune')
                        

                if args.add_cifar10:
                    logger.debug("Use cifar10 to distill again")
                    trainloader_cifar10, testloader_cifar10 = get_cifar10_loader(args)
                    run_concat_distill_multi(nets, cloud_net, args, trainloader_cifar10, testloader_public, worker_num=0, device=device, prefix='cifar10_')
                
                # if args.finetune:
                #     cloud_net = freeze_except_last_layer(cloud_net)
                #     run_train(cloud_net, 0, args, trainloader_public, testloader_public, 9, device, 'cloud_finetune')

            else:
                # non-iid here
                logger.debug(30*'*' + 'Non-iid' + 30*'*')

                logger.debug("Prepare cifar10 as public data")
                trainloader_cifar10, testloader_cifar10 = get_cifar10_loader(args)
                similarity_index = []

                for round in range(args.num_rounds):        
                    logger.debug(f"############# round {round} #############")
                    nets = check_model_trainable(nets)
                    for i in range(0, args.num_workers, 1):
                        run_train_non_iid(nets[i], round, args, trainloaders[i], testloader_non_iid[0], testloaders[i], i, device, 'edge_' + str(i))
                        
                        # loop over cifar10
                        # each image calculate similarity against each image in private dataset
                        # sum up the similarity and divided by the number of private images
                        # iterate over all the public data 
                        # calculate the total similarity
                        # similarity_index.append()

                    run_concat_distill_multi(nets, cloud_net, args, trainloader_all[0], testloader_non_iid[0], worker_num=0, device=device, prefix='distill_')

                exit()       
                
        logger.debug(30*'*' + 'Done training workers' + 30*'*')

        if not args.alternate:
            # run_distill(net_1, cloud_net, args, trainloader_1, testloader_1, worker_num=1, device=device)
            run_distill(nets[1], cloud_net, args, trainloader_1, testloader_1, worker_num=1, device=device)
            # test the first 10 classes
            logger.debug(30*'*' + 'Testing on the other 10 classes' + 30*'*')
            test_only(cloud_net, testloader, device)
        else:
            
            # use public to distill
            # need to add --public_distill flag
            if args.public_distill:

                # run_concat_distill_multi(net, net_1, net_2, cloud_net, args, trainloader_30cls_public, testloader_30cls, worker_num=0, device=device)
                logger.debug("Distilling with public data")
                
                # run_concat_distill_two(nets[0], nets[1], cloud_net, args, trainloader_public, testloader_public, worker_num=0, device=device)
                # Enable multiple edge models
                
                ######################### seperate distillation step #########################
                # run_concat_distill_multi(nets, cloud_net, args, trainloader_public, testloader_public, worker_num=0, device=device)
                ##############################################################################

                # try to use private data to distill
                # run_concat_distill_two(net, net_1, cloud_net, args, trainloader_private, testloader_public, worker_num=0, device=device)
            else:
                # if want to use private data to distill change the 'use_full_data' flag to false
                # concat dataset private
                # run_concat_distill_two(net[0], nets[1], cloud_net, args, trainloader_cloud, testloader_cloud, worker_num=0, device=device)
                # Enable multiple edge models
                run_concat_distill_multi(net[0], nets[1], cloud_net, args, trainloader_cloud, testloader_cloud, worker_num=0, device=device)

    else:
        logger.debug("Done experiment")
        exit()
