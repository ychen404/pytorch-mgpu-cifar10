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
import pdb


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
fmt_str = '%(levelname)s - %(message)s'
fmt_file = '[%(levelname)s]: %(message)s'

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('output.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter(fmt_str)
f_format = logging.Formatter(fmt_file)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


def parse_arguments():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--net', default='res8')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--bs', default='128')
    parser.add_argument('--workspace', default='')
    parser.add_argument("--split", default=0.5, type=float, help="split training data")
    parser.add_argument("--split_classes", action='store_true', help='split the number of classes to reduce difficulty')
    parser.add_argument('--temperature', default=5, type=float, help='temperature for distillation')
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--two', action='store_true')

    args = parser.parse_args()

    return args

def freeze_net(net):
    # freeze the layers of a model
    for param in net.parameters():
        param.requires_grad = False
    # set the teacher net into evaluation mode
    net.eval()
    return net

# Training
def train(epoch, net, criterion, optimizer, trainloader, device):
    logger.debug('\nEpoch: %d' % epoch)
    net.train()
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

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss/(batch_idx+1)


def test(epoch, net, criterion, testloader, device):

    best_acc = 0
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

            logger.debug(f"outputs.max(1): {outputs.max(1)}\n")
            logger.debug(f"predicted: {predicted}\n")
            logger.debug(f"targets: {targets}\n")
            total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            correct += torch.sum(predicted == targets.data).float()            
            logger.debug(f"correct={correct} / total {total}\n")
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logger.debug('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-ckpt.t7')

        best_acc = acc
    return acc, best_acc
        

def build_model_from_name(name, num_classes):
    if name =='res8':
        net = resnet8(num_classes=num_classes)    
    if name == 'res50':
        net = resnet50(num_classes=num_classes)
    if name == 'lenet':
        net = LeNet()
    elif name =='vgg':
        net = VGG('VGG19')

    print_total_params(net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True
    
    return net


def run_train(net, args, trainloader, testloader, device):
    
    list_loss = []
    strtime = get_time()
    criterion_edge = nn.CrossEntropyLoss()
    optimizer_edge = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch + args.epoch):
        trainloss = train(epoch, net, criterion_edge, optimizer_edge, trainloader, device)
        acc, best_acc = test(epoch, net, criterion_edge, testloader, device)
        logger.debug(f"The result is: {acc}")
        write_csv('acc_' + args.workspace + '_' + strtime + '.csv', str(acc))
        list_loss.append(trainloss)

    logger.debug("===> BEST ACC. PERFORMANCE: %.2f%%" % (best_acc))


def distill(epoch, edge_net, cluod_net, optimizer, trainloader, device, lambda_=1):
    
    logger.debug('\nEpoch: %d' % epoch)
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

        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)

        loss_kd = kd_fun(s_max, t_max) / batch_size
        loss = loss_fun(out_s, targets)

        loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

        loss_kd.backward()
        optimizer.step()

        After = list(cloud_net.parameters())[0].clone()
        logger.debug(torch.equal(Before.data, After.data))

        train_loss += loss_kd.item()
        logger.debug(f"loss_kd.item() {loss_kd.item()}\n")

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f ' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)


def run_distill(teacher, student, args, trainloader, testloader, worker_num, device):
    
    strtime = get_time()
    list_loss = []

    frozen_teacher = freeze_net(teacher)
    frozen_teacher = frozen_teacher.to(device)
    criterion_cloud = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epoch):
        
        trainloss = distill(epoch, frozen_teacher, student, optimizer, trainloader, device)
        acc, best_acc = test(epoch, student, criterion_cloud, testloader, device)

        logger.debug(f"The result is: {acc}")
        write_csv('distill_acc_' + args.workspace + '_' + 'worker_' + worker_num + '_' + strtime + '.csv', str(acc))
        list_loss.append(trainloss)

    logger.debug(f"Linear.weight after distillation from worker {worker_num}: {student.linear.weight.data}")
    logger.debug("===> BEST ACC. DISTILLATION: %.2f%%" % (best_acc))


    return student

if __name__ == "__main__":

    args = parse_arguments()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    bs = int(args.bs)

    # Data
    logger.debug('==> Preparing data..')
    num_classes = 10 if args.dataset == 'cifar10' else 100

    if args.dataset == 'cifar10':
        trainloader, testloader = get_cifar10_loader(args)

    else:
        logger.debug('==> CIFAR-100')
        trainset = get_cifar100()
        transform_test = get_cifar100_transfromtest()

        if args.split != 0 and not args.split_classes:
            logger.debug(f"Using {int(args.split * 100)}% of training data, classifying all classes")
            trainset_a, trainset_b = split_train_data(trainset, args.split)
            trainloader = torch.utils.data.DataLoader(trainset_a, batch_size=bs, shuffle=True, num_workers=4)
        
        elif args.split != 0 and args.split_classes:
            
            logger.debug(f"Using {int(args.split * 100)}% of training data and classifying {int(args.split * 100)}%")
            trainloader, testloader = get_worker_data(trainset, args, workerid=0)
            
            # two workers
            if args.two:    
                logger.debug(f"Creating a new loader Using {int(args.split * 100)}% of training data and classifying {int(args.split * 100)}%")               
                # workerid=1
                trainloader_1, testloader_1 = get_worker_data(trainset, args, workerid=1)

        else: 
            logger.debug(f"Using full training data")
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)

    # Model
    logger.debug('==> Building model..')

    net = build_model_from_name(args.net, num_classes)
    net_1 = build_model_from_name('res8', num_classes)
    cloud_net = build_model_from_name('res18', num_classes)
    cnet = build_model_from_name('res18', num_classes)


    if args.resume:
        load_checkpoint()
        
    # Train the first edge model
    run_train(net, args, trainloader, testloader, device)
    logger.debug(30*'*' + 'Before Distilling from worker 0' + 30*'*')
    print_param(cloud_net)

    # Distill from the first worker

    logger.debug(30*'*' + 'Distilling from worker 0' + 30*'*')
    run_distill(net, cloud_net, args, trainloader, testloader, worker_num=0, device=device)


    if args.two:
        
        # train the second edge model 
        logger.debug(f"Training worker 1")
        run_train(net_1, args, trainloader_1, testloader_1, device)

        # distill again
        logger.debug(30*'*' + 'Distilling from worker 1' + 30*'*')
        run_distill(net_1, cnet, args, trainloader_1, testloader_1, worker_num=1, device=device)
        
