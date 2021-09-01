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
from train_cifar import test

def build_model_from_name(name, num_classes):

    print(name, type(name))
    if name == 'res8':
        net = resnet8(num_classes=num_classes)    

    elif name == 'res20':
        net = resnet20(num_classes=num_classes)

    elif name == 'res18':
        net = resnet18(num_classes=num_classes)

    else:
        print('Not supported model')
        exit()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print_total_params(net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True
    
    return net

def test_only(
    net, 
    testloader,
    device
    )->None:
    """Test a model."""
    
    strtime = get_time()
    criterion = nn.CrossEntropyLoss()
    acc, best_acc = test(0, net, criterion, testloader, device, 'test')
    logger.debug(f"The accuracy is: {acc}")
    # write_csv('acc_' + args.workspace +  '_test_other_ten' + strtime + '.csv', str(acc))
    # write_csv('results/' + args.workspace, 'acc_' +  'test_other_ten' + strtime + '.csv', str(acc))
    # logger.debug("===> BEST ACCURACY: %.2f%%" % (best_acc))


def check_ensemble_accuracy(edge_net_0, edge_net_1, edge_net_2, testloader, device, average=False):

    best_acc = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs)
            
            if average:

                out_0 = edge_net_0(inputs)
                out_1 = edge_net_1(inputs)
                out_2 = edge_net_2(inputs)
                outputs = (out_0 + out_1 + out_2) / 3
            
            else:
                out_t_temp = []
                for input, target in zip(inputs, targets):
                    # need to use unsqueeze to change a 3-dimensional image to 4-dimensional
                    # [3,32,32] -> [1,3,32,32]
                    input = input.unsqueeze(0)
                    # pdb.set_trace()
                    if 0 <= target.item() <= 9 :
                        # logger.debug(f"edge 0\n")        
                        out_t_temp.append (edge_net_0(input))
                    elif 10 <= target.item() <= 19 :
                        # logger.debug(f"edge 1\n")        
                        out_t_temp.append (edge_net_1(input))
                    else:
                        # logger.debug(f"edge 2\n")        
                        out_t_temp.append (edge_net_2(input))

                outputs = torch.cat(out_t_temp, dim=0) # use dim 0 to stack the tensors
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            # pdb.set_trace()
            # what is output -2, -1, 0, 1?
            what, predicted = outputs.max(1)
            print(what.shape)
            print(predicted.shape)

            # logger.debug(f"outputs.max(1): {outputs.max(1)}\n")
            # logger.debug(f"predicted: {predicted}\n")
            # logger.debug(f"targets: {targets}\n")
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            logger.debug(f"correct={correct} / total {total}\n")
            acc = 100.*correct/total
    return acc

def distill(
    edge_net_0, 
    edge_net_1, 
    edge_net_2, 
    cloud_net, 
    trainloader_concat, 
    device,
    hard_logit=False,
    rmse=False,
    selection=False,
    use_pseudo_labels=False,
    lambda_=1,
    )->float:
    
    edge_net_0 = freeze_net(edge_net_0)
    edge_net_1 = freeze_net(edge_net_1)
    edge_net_2 = freeze_net(edge_net_2)

    cloud_net.train()
    logger.debug(f"model.train={cloud_net.training}")
    train_loss = 0
    correct = 0
    total = 0
    loss_fun = nn.CrossEntropyLoss()
    kd_fun = nn.KLDivLoss(reduction='sum')
    optimizer = optim.Adam(cloud_net.parameters(), lr=0.001, weight_decay=5e-4)
    T = 1

    counter = 0
    logger.debug(f"Lambda: {lambda_}")
    for batch_idx, (inputs, targets) in enumerate(trainloader_concat):
        inputs, targets = inputs.to(device), targets.to(device)
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

        if hard_logit:
            if rmse:
                logger.debug(f"Enable RMSE")
                criterion = nn.MSELoss()
                loss_kd = torch.sqrt(criterion(s_max, out_t)) / batch_size
            else:
                loss_kd = kd_fun(s_max, out_t) / batch_size
            
        else:
            loss_kd = kd_fun(s_max, t_max) / batch_size

        loss = loss_fun(out_s, targets)
        
        if use_pseudo_labels:
            logger.debug(f"Enable pseudo labels")
            loss_sd = loss_fun(out_s, pseudo_labels)
            # loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd + 0.5 * loss_sd
            loss_kd = loss_sd
        else:
            loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

        loss_kd.backward()
        optimizer.step()

        train_loss += loss_kd.item()
        
    return train_loss/(batch_idx+1), cloud_net

def freeze_net(net):
    # freeze the layers of a model
    for param in net.parameters():
        param.requires_grad = False
    # set the teacher net into evaluation mode
    net.eval()
    return net

def write_csv(workspace, path, content):
    fullpath = workspace + '/' + path
    with open(fullpath, 'a+') as f:
        f.write(content + '\n')

def save_model(net, msg):
    logger.debug('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    # torch.save(state, './checkpoint/' + args.net + '_' + msg +'_ckpt.t7')
    torch.save(state, './check_ensemble/' + msg +'_ckpt.t7')

def check_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)    



if __name__ == "__main__":

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Model
    logger.debug('==> Building model..')

    model = 'res8'
    num_classes = 100

    net = build_model_from_name(model, num_classes)
    net_1 = build_model_from_name(model, num_classes)
    net_2 = build_model_from_name(model, num_classes)
    cloud_net = build_model_from_name('res18', num_classes)
    criterion_cloud = nn.CrossEntropyLoss()

    net = load_edge_checkpoint(net, 'res8_edge_0_ckpt.t7')
    # net_1 = load_edge_checkpoint(net_1, 'res8_edge_1_ckpt.t7')
    net_1 = load_edge_checkpoint_fullpath(net_1, 'edge_checkpoint/res8_edge_1_ckpt.t7')
    net_2 = load_edge_checkpoint(net_2, 'res8_edge_2_ckpt.t7')

    testloader = torch.load('testloader_first_10cls.pth')
    testloader_1 = torch.load('testloader_second_10cls.pth')
    testloader_2 = torch.load('testloader_third_10cls.pth')
    testloader_20cls = torch.load('testloader_20cls.pth')
    testloader_30cls = torch.load('testloader_30cls.pth')
    trainloader_30cls = torch.load('trainloader_30cls.pth')

    test_only(net, testloader, 'cuda')
    test_only(net_1, testloader_1, 'cuda')
    test_only(net_2, testloader_2, 'cuda')

    acc = check_ensemble_accuracy(net, net_1, net_2, testloader_30cls, device, average=True)
    print(acc)
    exit()

    workspace = 'results/hard_logit_rmse'
    epochs = 200
    check_dir(workspace)

    for epoch in range(epochs):
        train_acc, cloud_net = distill(net, net_1, net_2, cloud_net, trainloader_30cls, device, hard_logit=True, rmse=True, selection=True, use_pseudo_labels=False)
        logger.debug(f'Train acc: {train_acc}')
        acc, best_acc = test(epoch, cloud_net, criterion_cloud, testloader_30cls, device, 'cloud')       
        write_csv(workspace, 'distill_train_loss' + '.csv', str(train_acc))
        write_csv(workspace, 'distill_test_acc' + '.csv', str(acc))

    save_model(cloud_net, 'test')