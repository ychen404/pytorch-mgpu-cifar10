# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch.nn as nn
import torch.nn.init as init
import torch
import logging
import datetime
from models import *
import torch.backends.cudnn as cudnn
from plot_results import *


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
fmt_str = '%(levelname)s - %(message)s'
fmt_file = '[%(levelname)s]: %(message)s'

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def load_checkpoint(net):

    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net


def load_edge_checkpoint(net, checkpoint):

    print('==> Resuming from checkpoint..')
    assert os.path.isdir('edge_checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./edge_checkpoint/' + checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net


def load_edge_checkpoint_fullpath(net, checkpoint):

    print('==> Resuming from checkpoint..')
    assert os.path.isdir('edge_checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net


def print_total_params(net):
    total_params = sum(p.numel() for p in net.parameters())
    layers = len(list(net.modules()))
    print(f" total parameters: {total_params}, layers {layers}")

def write_csv(workspace, path, content):
    fullpath = workspace + '/' + path
    with open(fullpath, 'a+') as f:
        f.write(content + '\n')

def print_param(model):
    for name, param in model.named_parameters():
        # if param.requires_grad:
        print (f"{name}, \t\trequires_grad={param.requires_grad}")

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def get_time():
    
    strtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    return strtime

def get_logger_handler(path):
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(path)
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter(fmt_str)
    f_format = logging.Formatter(fmt_file)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    return c_handler, f_handler

def check_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)    


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


def save_figure(path, csv_name):

    # pdb.set_trace()
    data_to_plot = {}
    data_to_plot['data'] = collect_data(path + '/' + csv_name)
    print(data_to_plot)
    plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Accuracy', output= path + '/' + 'result.png')

def build_model_from_name(name, num_classes, device):

    # print(name, type(name))
    # print(name == 'res8')

    if name == 'res6':
        net = resnet6(num_classes=num_classes)

    elif name == 'res6_emb':
        net = resnet6(num_classes=num_classes, emb=True)

    elif name == 'res8':
        net = resnet8(num_classes=num_classes)    

    elif name == 'res50':
        net = resnet50(num_classes=num_classes)

    elif name == 'res20':
        net = resnet20(num_classes=num_classes)
    
    elif name == 'res20_emb':
        net = resnet20(num_classes=num_classes, emb=True)

    elif name == 'res18':
        net = resnet18(num_classes=num_classes)
    
    elif name == 'lenet':
        net = LeNet()

    elif name =='vgg':
        net = VGG('VGG19')

    else:
        NotImplementedError('Not supported model')
        exit()

    print_total_params(net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

    return net

def save_figure(path, csv_name):

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


def count_targets(targets):
    
    """
    Count the number of samples to determine the weights
    """
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