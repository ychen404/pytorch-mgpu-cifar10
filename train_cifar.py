# -*- coding: utf-8 -*-
'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function
from xml.dom import NotSupportedErr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging

from models import *
from utils import *
from data_loader import *
import pdb
from plot_results import *
import random
from copy import deepcopy


logger = logging.getLogger('__name__')
logger.setLevel('DEBUG')
# fmt_str = '%(levelname)s - %(message)s'
# fmt_file = '[%(levelname)s]: %(message)s'

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
    parser.add_argument('--partition_mode', default='dirichlet')
    parser.add_argument('--public_distill', action='store_true', help="use public data to distill")
    parser.add_argument('--exist_loader', action='store_true', help="there is exist loader")
    parser.add_argument('--save_loader', action='store_true', help="save trainloaders")
    parser.add_argument('--alpha', default=100, type=float, help='alpha for dirichle partition')
    parser.add_argument('--lamb', default=0.5, type=float, help='lambda for distillation')
    parser.add_argument('--temp', default=1, type=float, help='temperature for distillation')
    parser.add_argument('--public_percent', default=0.5, type=float, help='percentage training data to be public')
    parser.add_argument('--distill_percent', default=1, type=float, help='percentage of public data use for distillation')
    parser.add_argument('--vary_epoch', action='store_true', help="change the number of local epochs of edges")

    ######################### Aggregation parameters #########################
    parser.add_argument('--selection', action='store_true', help="enable selection method")
    parser.add_argument('--dlc', action='store_true', help="enable selection method")
    parser.add_argument('--num_drop', default=1, type=int, help='number of edges to be dropped')
    parser.add_argument('--finetune', action='store_true', help="finetune the cloud model") # test fine-tune
    parser.add_argument('--finetune_epoch', default=10, type=int, help='number of epochs for finetune')
    parser.add_argument('--finetune_percent', default=0.2, type=float, help='percentage of data to finetune')
    parser.add_argument('--sample_percent', default=1, type=float, help='percentage of private data in each worker')

    args = parser.parse_args()

    return args


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
        
        if net.emb:
            outputs, emb = net(inputs)
        else:
            outputs = net(inputs)
        # pdb.set_trace()
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
            
            # Check if emb attribute exists
            # emb only exists in edge models 
            if hasattr(net, 'emb') and net.emb:
                outputs, emb = net(inputs)
            else:
                outputs = net(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)

            logger.debug(f"predicted: {predicted}\n")
            logger.debug(f"targets: {targets}\n")
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            logger.debug(f"correct={correct} / total {total}\n")


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
        

def run_train(net, round, args, trainloader, testloader, testloader_local, worker_num, device, vary_epoch, mode, msg):
    """
    This is the wrapper function that calls train() and test() for training all the edge models.  
    Merge this with run_train()
    The only difference is performing extra step for local classes accuracy
    Use 'mode' to differentiate save local accuracy or not

    mode = local -> save local accuracy
    model = None -> don't save local accuracy
    """
    
    list_loss = []
    strtime = get_time()
    criterion_edge = nn.CrossEntropyLoss()
    optimizer_edge = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Save csv files during training
    csv_name = 'acc_'  + 'worker_' + str(worker_num) + '_' + strtime + '.csv'
    csv_name_local = 'acc_'  + 'worker_' + str(worker_num) + '_' + strtime + '_' + 'local' + '.csv'
    path = 'results/' + args.workspace
        
    # Simple test (remove later)
    if vary_epoch:
        args.epoch = 1

    # for epoch in range(start_epoch, start_epoch + args.epoch):   
    for epoch in range(args.epoch):     
        trainloss = train(epoch, round, net, criterion_edge, optimizer_edge, trainloader, device)
        acc, best_acc = test(epoch, args, net, criterion_edge, testloader, device, msg)
        logger.debug(f"The result is: {acc}")
        write_csv(path, csv_name, str(acc))

        if mode == 'local':
            acc_local, _ = test(epoch, args, net, criterion_edge, testloader_local, device, msg)
            write_csv(path, csv_name_local, str(acc_local))

        list_loss.append(trainloss)

    # pdb.set_trace()
    # save_figure(path, csv_name)
    
    logger.debug("===> BEST ACC. PERFORMANCE: %.2f%%" % (best_acc))


def distill(
    epoch,  
    edge_nets, 
    cloud_net, 
    optimizer, 
    trainloader_concat, 
    device,
    num_workers,
    split,
    distill_percent, 
    dataset,
    average_method,
    select_mode,
    drop_leastconfident,
    num_drop,
    selection=False,
    lambda_=1,
    temperature_=1,
    )->float: 
    
    logger.debug('\nEpoch: %d' % epoch)

    def return_edge_norm(edge_out, emb):

        nLab = total_classes
        batchProbs = F.softmax(edge_out, dim=1).data.cpu().numpy()
        maxInds = np.argmax(batchProbs,1)
        size = 1
        
        embedding = np.zeros([size, embDim * nLab])
        idxs = np.arange(size)
        for j in range(size):
            for c in range(nLab):
                if c == maxInds[j]:
                    embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(emb[j]) * (1 - batchProbs[j][c])
                else:
                    embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(emb[j]) * (-1 * batchProbs[j][c])

        emb_norm = np.linalg.norm(embedding, 2)

        return emb_norm
    

    cloud_net.train()
    # logger.debug(f"model.train={cloud_net.training}")
    train_loss = 0
    loss_fun = nn.CrossEntropyLoss()
    kd_fun = nn.KLDivLoss(reduction='sum')
    T = temperature_

    # consider cifar100 only for now
    if dataset == 'cifar100':
        total_classes = 100 
    elif dataset == 'cifar10':
        total_classes = 10
    else:
        raise NotImplementedError("Not supported dataset")

    worker_ids = [x for x in range(num_workers)]
    num_classes = int(split * total_classes)
    bounds = []
    for worker_id in worker_ids:
        start = worker_id * num_classes + 0
        end = worker_id * num_classes + num_classes - 1
        bounds.append([start, end])

    counter = 0
    # logger.debug(f"Lambda: {lambda_}")
    for batch_idx, (inputs, targets) in enumerate(trainloader_concat):
        inputs, targets = inputs.to(device), targets.to(device)
        # logger.debug(f"device = {device}")
        optimizer.zero_grad()
        counter += 1
        # logger.debug(f"Counter: {counter}")
        out_s = cloud_net(inputs)

        size_curr_batch = out_s.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)

        out_t_temp = []

        if selection: # direct the data to edge worker based on classes 
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
                    raise NotImplementedError("Not support weighted average now")
                    
                else:
                    logger.debug("No select model provided")
                    
            logger.debug(f"len out_t_temp: {len(out_t_temp)}")
            out_t = torch.cat(out_t_temp, dim=0) # use dim 0 to stack the tensors
            assert out_t.shape[1] == 100, f"the shape is {out_t.shape}, should be (x, 100)"
            t_max = F.softmax(out_t / T, dim=1)
        
        else:
            # Use torch.zeros(128,100) to create the placeholder
            out_t = torch.zeros((size_curr_batch, total_classes), device=device)
            t_max = torch.zeros((size_curr_batch, total_classes), device=device)
            
            # create place holder for emb method
            batch_memory = [-1] * size_curr_batch
            norm_idx = 1
            output_idx = 2

            save_all_output = [[] for _ in range(len(inputs))]

            
            # The output size of each sample is torch.Size([1, 10]) for cifar10
            num_left_edge = len(edge_nets) - num_drop
            temp_res = torch.empty((num_left_edge, total_classes), device=device)

            emb_flag = False
            # pdb.set_trace()
            # TODO: Need to add a method to distinguish with other
            if drop_leastconfident:
                emb_flag = True
                for idx, (input, target) in enumerate(zip(inputs, targets)):
                    input = input.unsqueeze(0)

                    for i, edge_net in enumerate(edge_nets):
                        # logger.debug(f"Processing the {i} edge")

                        if hasattr(edge_net, 'emb') and edge_net.emb:

                            edge_out, emb = edge_net(input)
                            embDim = edge_net.get_embedding_dim()
                            emb = emb.data.cpu().numpy()
                            emb_norm = return_edge_norm(edge_out, emb)
                            
                            save_all_output[idx].append((emb_norm, edge_out))

                    sorted_output = save_all_output[idx] # save the outputs from a batch 
                    sorted_output.sort(key = lambda x : x[0]) # use the emb_norm to sort
                    
                    for i, e in enumerate(sorted_output[:-num_drop]):
                        temp_res[i] = e[1] # the 1st idx in e is the model output
                    
                    # Calculate the average of the output of each sample
                    # Each output is torch.size([1,10])
                    # Use dim=0, the size of the mean is torch.size([10]), use keepdim to maintain the torch.size([1, 10])

                    out_t_temp.append(temp_res.mean(dim=0, keepdim=True))
                
                out_t = torch.cat(out_t_temp, dim=0) # use dim 0 to stack the tensors
                # logger.debug(f"out_t shape: {out_t.shape}")

            #### Calculate model outputs from all edge models without dropping
            else:
                for i, edge_net in enumerate(edge_nets):
                    edge_net = edge_net.to(device)
                    out_t += edge_net(inputs)

                # logger.debug(f"The edge_net.emb flag is {edge_net.emb}")
                
                # if hasattr(edge_net, 'emb') and edge_net.emb:
                #     emb_flag = True
                #     negative_one, replaced = 0, 0
                #     for idx, (input, target) in enumerate(zip(inputs, targets)):
                #         input = input.unsqueeze(0)
                #         edge_out, emb = edge_net(input)
                #         embDim = edge_net.get_embedding_dim()
                #         emb = emb.data.cpu().numpy()
                #         emb_norm = return_edge_norm(edge_out, emb)
                        
                #         pdb.set_trace()
                #         if drop_leastconfident: 

                #             # Used for drop least confidence
                #             # Create memory for each sample in each batch of each edge worker
                #             # Use the norm of the second to the last layer output as an indicator
                #             # 'num_drop' determines how many edge workers are dropping
                #             # We don't need to store 'i' in this case
                #             # But it's better to use consistent indexing
                #             save_all_output[idx].append((i, emb_norm, edge_out))

                #         else:
                #             if batch_memory[idx] == -1:
                #                 batch_memory[idx] = (i, emb_norm, edge_out, target)
                                
                #             else: 
                #                 if  emb_norm < batch_memory[idx][norm_idx]:
                #                     batch_memory[idx] = (i, emb_norm, edge_out, target)
                #                     replaced += 1


            # if emb_flag:
            #     logger.debug(f"emb mode")
            #     stack_out = []
            #     dropped = []
            #     if drop_leastconfident:
            #         pass
            #     else:    
            #         for b in batch_memory:
            #             stack_out.append(b[output_idx])

            #         not_replaced = len(batch_memory) - replaced
            #         logger.debug(f"Not replaced: {not_replaced}, replaced: {replaced}, not_replaced/total: {not_replaced/len(batch_memory)}")

            #     out_t = torch.cat(stack_out, dim=0) # use dim 0 to stack the tensors

            t_max += F.softmax(out_t / T, dim=1)
            
            if average_method == 'equal':
                logger.debug("Equal weights")
                t_max = t_max / num_workers
            
            elif average_method == 'grad':
                # Use gradient magnitude to select workers
                # So nothing needs to be done here for now.
                # We may need to add some additional operations 
                # logger.debug("Grad mode")
                pass

            else: 
                raise NotImplementedError("Not support weighted average now")

        loss_kd = kd_fun(s_max, t_max) / size_curr_batch
        loss = loss_fun(out_s, targets)
        
        loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

        loss_kd.backward()
        optimizer.step()

        train_loss += loss_kd.item()
        
    return train_loss/(batch_idx+1)


def run_distill(
    edge_nets,  
    cloud, 
    args, 
    trainloader_cloud, 
    testloader_cloud,
    worker_num, 
    device,
    selection,
    prefix
    )->nn.Module:
    
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
        
        # TODO: merge selection into average method
        trainloss = distill(epoch,
                            frozen_edge_nets,
                            cloud, 
                            optimizer, 
                            trainloader_cloud, 
                            device, 
                            args.num_workers,
                            args.split,
                            args.distill_percent,
                            dataset=args.dataset,
                            average_method='grad', 
                            select_mode='guided',
                            drop_leastconfident=args.dlc,
                            num_drop=args.num_drop,
                            selection=selection, 
                            lambda_=args.lamb,
                            temperature_=args.temp)

        acc, best_acc = test(epoch, args, cloud, criterion_cloud, testloader_cloud, device, 'cloud')

        logger.debug(f"The result is: {acc}")
        # write_csv('results/' + args.workspace, 'distill_concat_' + strtime + '.csv', str(acc))
        write_csv('results/' + args.workspace, prefix + strtime + '.csv', str(acc))

        list_loss.append(trainloss)

    logger.debug("===> BEST ACC. DISTILLATION: %.2f%%" % (best_acc))

    return cloud


if __name__ == "__main__":

    args = parse_arguments()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    torch.manual_seed(0)
    print(args)

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
        # trainloader, testloader = get_cifar10_loader(args, data_only=False)
        logger.info('==> CIFAR-10')
        trainset, testset = get_cifar10()

    elif args.dataset == 'cifar100':
        logger.info('==> CIFAR-100')
        trainset, testset = get_cifar100()
        transform_test = get_cifar100_transfromtest()
        # trainset, testset = get_cifar10()
    else:
        raise NotImplementedError("Not supported dataset")

        # trainset, testset = get_cifar10_loader(args, data_only=True)

    if args.public_distill: 
        trainset_public, trainset_private = split_train_data(trainset, args.public_percent)

        # Use 10% of the public dataset
        if args.distill_percent != 1:
            # do notthing for now (moved insided the distillation function)
            logger.info(f"The distill percent is {args.distill_percent}")

    if args.split != 0 and not args.split_classes:
        logger.info(f"Using {int(args.split * 100)}% of training data, classifying all classes")
        trainset_a, trainset_b = split_train_data(trainset, args.split)
        trainloader = torch.utils.data.DataLoader(trainset_a, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    elif args.split != 0 and args.split_classes and not args.baseline:
        # Use all data
        # logger.info(f"is it here Using {int(args.split * 100)}% of training data and classifying {int(args.split * 100)}%")
        logger.info(f"is it here Using {int(args.split * 100)}% of training data and classifying {int(args.split * 100)}%")

        if args.exist_loader:
            trainloader = torch.load('trainloader_first_10cls.pth')
            testloader = torch.load('testloader_first_10cls.pth')
        
        else:
            client_classes = int(num_classes * args.split) 
            logger.info(f"client classes: {client_classes}")
            
            # Use public data to perform distillation
            if args.public_distill:
                
                # Split the classes uniformly. 
                # The first worker has classes [0,1], the second worker has classes [2,3] etc
                if args.partition_mode == 'uniform':
                    trainloaders = get_subclasses_loaders(trainset_private, args.num_workers, client_classes, num_workers=4, seed=100)
                    class_select = args.num_workers * client_classes

                if args.partition_mode == 'dirichlet':
                    
                    s = args.split if args.split == 1 else args.split * args.num_workers
                    extract_trainset = extract_classes(trainset_private, s, dataset=args.dataset, workerid=0)
                    
                    # use 1 thread worker instead of 4 in the single gpu case
                    trainloaders = get_dirichlet_loaders(extract_trainset, n_clients=args.num_workers, alpha=args.alpha, num_workers=1, seed=100)
                    class_select = client_classes * args.num_workers

                logger.info(f"Cloud Test data")
                testloader_cloud = get_subclasses_loaders(testset, n_clients=1, client_classes=class_select, num_workers=4, seed=100)

                logger.info(f"Edge Test data")
                
                # use all the test data
                # no need to use the get_subclasses_loaders function
                # TODO: Is it still necessary to make multiple copies of the testloader
                if args.split == 1: 
                    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
                    testloaders = [testloader for _ in range(args.num_workers)]
                else:
                    testloaders = get_subclasses_loaders(testset, args.num_workers, client_classes, num_workers=4, seed=100)
                
                logger.info(f"Cloud Train loader")
                trainloader_cloud = get_subclasses_loaders(trainset_public, n_clients=1, client_classes=args.num_workers * client_classes, num_workers=4, seed=100)

            # Use private data to perform distillation       
            else:
                # trainloader, testloader = get_worker_data(trainset, args, workerid=0)
                trainloaders = get_subclasses_loaders(trainset, args.num_workers, client_classes, num_workers=4, seed=100)
                trainloader_cloud = get_subclasses_loaders(trainset, n_clients=1, client_classes=int(args.num_workers * client_classes), num_workers=4, seed=100)

                # _, testloader_non_iid = get_worker_data_hardcode(trainset, args.split, workerid=0)
                testloader_cloud = get_subclasses_loaders(testset, n_clients=1, client_classes=int(args.num_workers * client_classes), num_workers=4, seed=100)

                # logger.debug(testloader_non_iid[0])
                testloaders = get_subclasses_loaders(testset, args.num_workers, client_classes, num_workers=4, seed=100)

        if args.save_loader:
            torch.save(trainloader, 'trainloader_first_10cls.pth')
            torch.save(testloader, 'testloader_first_10cls.pth')


    # use to collect baseline results
    # TODO: simplify the test cases. The test cases are overlapped. 
    
    #################################### For baseline test ####################################
    elif args.split != 0 and args.baseline:
        logger.info(f"Using {int(args.split * 100)}% for simple baseline")
        trainloader, testloader = get_worker_data(trainset, args, workerid=0)

    ###########################################################################################

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

    if args.resume:
        # load edge checkpoints
        ckpt_paths = ['results/edge_ckpts/edge_0_ckpt.t7', 'results/edge_ckpts/edge_1_ckpt.t7']
        for i in range(args.num_workers):
            nets[i] = load_edge_checkpoint_fullpath(nets[i], ckpt_paths[i])

    for round in range(args.num_rounds):        
        logger.debug(f"############# round {round} #############")
        nets = check_model_trainable(nets)
        for i in range(args.num_workers):
            if not args.resume:
                if args.vary_epoch:
                    if i == 0: # test training only the first edge
                        logger.debug(f"{args.num_workers}, {i}")
                        run_train(nets[i], round, args, trainloaders[i], testloader_cloud, testloaders[i], i, device, False, 'local', 'edge_' + str(i))
                        logger.debug("Done edge training")
                    else:
                        run_train(nets[i], round, args, trainloaders[i], testloader_cloud, testloaders[i], i, device, True, 'local', 'edge_' + str(i))
                else:
                    run_train(nets[i], round, args, trainloaders[i], testloader_cloud, testloaders[i], i, device, False, 'local', 'edge_' + str(i))
                                    
        # Get the public data with the specified classes
        run_distill(nets, 
                    cloud_net, 
                    args, 
                    trainloader_cloud, 
                    testloader_cloud, 
                    worker_num=0, 
                    device=device,
                    selection=args.selection, 
                    prefix='distill_')
