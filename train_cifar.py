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
    parser.add_argument('--cloud_lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--cloud_epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--net', type=str, default='res8')
    parser.add_argument('--cloud', type=str, default='res18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--workspace', default='')
    parser.add_argument("--split", default=0.5, type=float, help="split training data")
    parser.add_argument("--split_classes", action='store_true', help='split the number of classes to reduce difficulty')
    parser.add_argument("--baseline", action='store_true', help='Perform only training for collection baseline')
    parser.add_argument('--temperature', default=5, type=float, help='temperature for distillation')
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--two', action='store_true')
    parser.add_argument('--alternate', action='store_true', help="test the alternative case")
    parser.add_argument('--exist_loader', action='store_true', help="there is exist loader")
    parser.add_argument('--save_loader', action='store_true', help="save trainloaders")



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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
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

    print(name, type(name))
    print(name == 'res8')

    if name == 'res8':
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


def run_train(net, args, trainloader, testloader, worker_num, device):
    
    list_loss = []
    strtime = get_time()
    criterion_edge = nn.CrossEntropyLoss()
    optimizer_edge = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch + args.epoch):
        trainloss = train(epoch, net, criterion_edge, optimizer_edge, trainloader, device)
        acc, best_acc = test(epoch, net, criterion_edge, testloader, device)
        logger.debug(f"The result is: {acc}")
        # write_csv('acc_' + args.workspace + '_worker_' + str(worker_num) + '_' + strtime + '.csv', str(acc))
        write_csv('results/' + args.workspace, 'acc_'  + 'worker_' + str(worker_num) + '_' + strtime + '.csv', str(acc))

        list_loss.append(trainloss)

    logger.debug("===> BEST ACC. PERFORMANCE: %.2f%%" % (best_acc))


def distill(epoch, edge_net, cloud_net, optimizer, trainloader, device, lambda_=1):
    
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

def distill_from_two_workers(epoch, edge_net_0, edge_net_1, cloud_net, optimizer, trainloader_0, trainloader_1, device, lambda_=1):
    
    logger.debug('\nEpoch: %d' % epoch)
    cloud_net.train()
    logger.debug(f"model.train={cloud_net.training}")
    train_loss = 0
    correct = 0
    total = 0
    loss_fun = nn.CrossEntropyLoss()
    kd_fun = nn.KLDivLoss(reduction='sum')
    T = 1

    trainloaders = [trainloader_0, trainloader_1]
    logger.debug(f"Lenth trainloaders, 0:{len(trainloader_0)}, 1:{len(trainloader_1)}")

    counter = 0
    # pack two loaders together and alternate the images 
    for trainloader in itertools.zip_longest(*trainloaders):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
    # for batch_idx, data in enumerate(itertools.chain(*trainloaders)):
        # inputs = data[0]
        # targets = data[1]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            counter += 1
            logger.debug(f"Counter: {counter}")
            out_s = cloud_net(inputs)
            # Use the knowledge from the correct edge model
            # Only check the first sample is enough        
            if 0 <= targets[0].item() <= 9:
                logger.debug(f"Batch idx: {batch_idx}, Worker 0 data")
                out_t = edge_net_0(inputs)
            elif 10 <= targets[0].item() <= 19:
                logger.debug(f"Batch idx: {batch_idx}, Worker 1 data")
                out_t = edge_net_1(inputs)
            else:
                logger.debug(f"Batch idx: {batch_idx}, Worker 2 data")
                out_t = edge_net_1(inputs)


            batch_size = out_s.shape[0]

            s_max = F.log_softmax(out_s / T, dim=1)
            t_max = F.softmax(out_t / T, dim=1)

            loss_kd = kd_fun(s_max, t_max) / batch_size
            loss = loss_fun(out_s, targets)

            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

            loss_kd.backward()
            optimizer.step()

            train_loss += loss_kd.item()
            logger.debug(f"loss_kd.item() {loss_kd.item()}\n")

            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f ' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)

def distill_from_multi_workers(
    epoch, 
    edge_net_0, 
    edge_net_1, 
    edge_net_2, 
    cloud_net, 
    optimizer, 
    trainloader_0, 
    trainloader_1, 
    trainloader_2, 
    device, 
    lambda_=1
    )->float:
    
    logger.debug('\nEpoch: %d' % epoch)
    cloud_net.train()
    logger.debug(f"model.train={cloud_net.training}")
    train_loss = 0
    correct = 0
    total = 0
    loss_fun = nn.CrossEntropyLoss()
    kd_fun = nn.KLDivLoss(reduction='sum')
    T = 1

    trainloaders = [trainloader_0, trainloader_1, trainloader_2]
    logger.debug(f"Lenth trainloaders, 0:{len(trainloader_0)}, 1:{len(trainloader_1)}, 2:{len(trainloader_2)}")

    counter = 0
    # pack two loaders together and alternate the images 
    for trainloader in itertools.zip_longest(*trainloaders):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
    # for batch_idx, data in enumerate(itertools.chain(*trainloaders)):
        # inputs = data[0]
        # targets = data[1]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            counter += 1
            logger.debug(f"Counter: {counter}")
            out_s = cloud_net(inputs)
            # Use the knowledge from the correct edge model
            # Only check the first sample is enough        
            if 0 <= targets[0].item() <= 9:
                logger.debug(f"Batch idx: {batch_idx}, Worker 0 data")
                out_t = edge_net_0(inputs)
            elif 10 <= targets[0].item() <= 19:
                logger.debug(f"Batch idx: {batch_idx}, Worker 1 data")
                out_t = edge_net_1(inputs)
            elif 20 <= targets[0].item() <= 29:
                logger.debug(f"Batch idx: {batch_idx}, Worker 2 data")
                out_t = edge_net_2(inputs)

            batch_size = out_s.shape[0]

            s_max = F.log_softmax(out_s / T, dim=1)
            t_max = F.softmax(out_t / T, dim=1)

            loss_kd = kd_fun(s_max, t_max) / batch_size
            loss = loss_fun(out_s, targets)

            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

            loss_kd.backward()
            optimizer.step()

            train_loss += loss_kd.item()
            # logger.debug(f"loss_kd.item() {loss_kd.item()}\n")

            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f ' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)


def run_alternate_distill(
    edge_0,
    edge_1,
    cloud,
    args,
    trainloader_0,
    trainloader_1,
    testloader_20cls,
    worker_num,
    device
    )->nn.Module:
    """Alternate data from two trainloaders to distill"""
    
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
        optimizer = optim.Adam(cloud.parameters(), lr=args.cloud_lr, weight_decay=5e-4)

    for epoch in range(args.epoch):
        
        # trainloss = distill(epoch, frozen_edge, cloud, optimizer, trainloader, device)
        # acc, best_acc = test(epoch, cloud, criterion_cloud, testloader, device)
        trainloss = distill_from_two_workers(epoch, frozen_edge_0, frozen_edge_1, cloud, optimizer, trainloader_0, trainloader_1, device)
        acc, best_acc = test(epoch, cloud, criterion_cloud, testloader_20cls, device)

        logger.debug(f"The result is: {acc}")
        write_csv('results/' + args.workspace, 'distill_alternate_' + strtime + '.csv', str(acc))
        list_loss.append(trainloss)

    logger.debug("===> BEST ACC. DISTILLATION: %.2f%%" % (best_acc))

    return cloud

def run_alternate_distill_multi(
    edge_0, 
    edge_1, 
    edge_2,
    cloud, 
    args, 
    trainloader_0, 
    trainloader_1,
    trainloader_2,
    testloader_30cls,
    worker_num, 
    device
    )->nn.Module:
    """Alternate data from two trainloaders to distill"""
    
    logger.debug("From multiple workers")
    strtime = get_time()
    list_loss = []

    frozen_edge_0 = freeze_net(edge_0)
    frozen_edge_0 = frozen_edge_0.to(device)
    
    frozen_edge_1 = freeze_net(edge_1)
    frozen_edge_1 = frozen_edge_1.to(device)

    frozen_edge_2 = freeze_net(edge_2)
    frozen_edge_2 = frozen_edge_2.to(device)
    
    criterion_cloud = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(cloud.parameters(), lr=args.cloud_lr, momentum=0.9, weight_decay=5e-4)
    else:
        logger.debug("Using adam optimizer")
        optimizer = optim.Adam(cloud.parameters(), lr=args.cloud_lr, weight_decay=5e-4)

    for epoch in range(args.cloud_epoch):
        
        trainloss = distill_from_multi_workers(epoch, frozen_edge_0, frozen_edge_1, frozen_edge_2, cloud, optimizer, trainloader_0, trainloader_1, trainloader_2, device)
        acc, best_acc = test(epoch, cloud, criterion_cloud, testloader_30cls, device)

        logger.debug(f"The result is: {acc}")
        write_csv('results/' + args.workspace, 'distill_alternate_' + strtime + '.csv', str(acc))
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
        
        trainloss = distill(epoch, frozen_edge, cloud, optimizer, trainloader, device)
        acc, best_acc = test(epoch, cloud, criterion_cloud, testloader, device)

        logger.debug(f"The result is: {acc}")
        # write_csv('distill_acc_' + args.workspace + '_' + 'worker_' + str(worker_num) + '_' + strtime + '.csv', str(acc))
        write_csv('results/' + args.workspace, 'worker_' + str(worker_num) + '_' + strtime + '.csv', str(acc))

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
    acc, best_acc = test(0, net, criterion, testloader, device)
    logger.debug(f"The result is: {acc}")
    # write_csv('acc_' + args.workspace +  '_test_other_ten' + strtime + '.csv', str(acc))
    write_csv('results/' + args.workspace, 'acc_' +  'test_other_ten' + strtime + '.csv', str(acc))
    logger.debug("===> BEST ACC. ON OTHER TEN CLASSES: %.2f%%" % (best_acc))

def check_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
if __name__ == "__main__":

    args = parse_arguments()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    torch.manual_seed(0)

    check_dir('results/' + args.workspace)
    
    # Data
    logger.debug('==> Preparing data..')
    num_classes = 10 if args.dataset == 'cifar10' else 100

    if args.dataset == 'cifar10':
        trainloader, testloader = get_cifar10_loader(args)

    else:
        logger.info('==> CIFAR-100')
        trainset = get_cifar100()
        transform_test = get_cifar100_transfromtest()

        if args.split != 0 and not args.split_classes:
            logger.info(f"Using {int(args.split * 100)}% of training data, classifying all classes")
            trainset_a, trainset_b = split_train_data(trainset, args.split)
            trainloader = torch.utils.data.DataLoader(trainset_a, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        elif args.split != 0 and args.split_classes:
            
            logger.info(f"Using {int(args.split * 100)}% of training data and classifying {int(args.split * 100)}%")
            
            # Use saved loader to save time
            # exist_loader = False
            
            
            if args.exist_loader:
                trainloader = torch.load('trainloader_first_10cls.pth')
                testloader = torch.load('testloader_first_10cls.pth')

            else:
                trainloader, testloader = get_worker_data(trainset, args, workerid=0)
            
            # save_loader = True # Save the loaders
            
            logger.debug(f"exist_loader: {args.exist_loader}, save_loader: {args.save_loader}")

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

                    logger.debug("Done loading loaders")

                else:
                    logger.debug("In the else condition")
                    trainloader_1, testloader_1 = get_worker_data(trainset, args, workerid=1)
                    trainloader_2, testloader_2 = get_worker_data(trainset, args, workerid=2)
                    # get a test loader for the first 20 classes
                    _, testloader_20cls = get_worker_data_hardcode(trainset, 0.2, workerid=0)
                    _, testloader_20cls_disjoint = get_worker_data_hardcode(trainset, 0.2, workerid=0, disjoint=True)
                    _, testloader_30cls = get_worker_data_hardcode(trainset, 0.3, workerid=0)

                if args.save_loader:
                    torch.save(trainloader_1, 'trainloader_second_10cls.pth')
                    torch.save(testloader_1, 'testloader_second_10cls.pth')

                    torch.save(trainloader_2, 'trainloader_third_10cls.pth')
                    torch.save(testloader_2, 'testloader_third_10cls.pth')
                    
                    torch.save(testloader_20cls_disjoint, 'testloader_20cls_disjoint.pth')
                    torch.save(testloader_20cls, 'testloader_20cls.pth')
                    torch.save(testloader_30cls, 'testloader_30cls.pth')
                    logger.debug("Done saving loaders")
                    exit()

        # use to collect baseline results
        elif args.split != 0 and args.baseline:
            logger.info(f"Using {int(args.split * 100)}% for baseline")
            trainloader, testloader = get_worker_data(trainset, args, workerid=0)

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

    net = build_model_from_name(args.net, num_classes)
    net_1 = build_model_from_name(args.net, num_classes)
    net_2 = build_model_from_name(args.net, num_classes)

    cloud_net = build_model_from_name(args.cloud, num_classes)
    # cloud_net = build_model_from_name('res50', num_classes)

    if args.resume:
        load_checkpoint()
        
    # Train the first edge model
    run_train(net, args, trainloader, testloader, 0, device)
    logger.debug(30*'*' + 'Before Distilling from worker 0' + 30*'*')
    
    if args.baseline:
        exit()

    logger.debug(f"*********Cloud param*********")
    print_param(cloud_net)

    logger.debug(f"*********Edge param*********")
    print_param(net)

    # Distill from the first worker
    logger.debug(30*'*' + 'Distilling from worker 0' + 30*'*')

    if not args.alternate: # if not alternate, then perform sequential distillation
        run_distill(net, cloud_net, args, trainloader, testloader, worker_num=0, device=device)

    logger.debug(f"*********Edge param after*********")
    print_param(net)

    if args.two:

        # run_train(net_1, args, trainloader_1, testloader_1, 1, device)
        run_train(net_2, args, trainloader_2, testloader_2, 2, device)

        logger.debug(30*'*' + 'Cloud param' + 30*'*')
        print_param(cloud_net)

        logger.debug(30*'*' + 'Edge param' + 30*'*')
        print_param(net_1)

        logger.debug(30*'*' + 'Distilling from worker 1' + 30*'*')

        if not args.alternate:
            # run_distill(net_1, cloud_net, args, trainloader_1, testloader_1, worker_num=1, device=device)
            run_distill(net_1, cloud_net, args, trainloader_1, testloader_1, worker_num=1, device=device)
            # test the first 10 classes
            logger.debug(30*'*' + 'Testing on the other 10 classes' + 30*'*')
            test_only(cloud_net, testloader, device)
        else:
            
            # Continual two classes
            # run_alternate_distill(net, net_1, cloud_net, args, trainloader, trainloader_1, testloader_20cls, worker_num=0, device=device)
            
            # test with the other two classes (the cloud is having trouble again)
            run_alternate_distill(net, net_2, cloud_net, args, trainloader, trainloader_2, testloader_20cls, worker_num=0, device=device)

            # with 3 workers
            # run_alternate_distill_multi(net, net_1, net_2, cloud_net, args, trainloader, trainloader_1, trainloader_2, testloader_30cls, worker_num=0, device=device)