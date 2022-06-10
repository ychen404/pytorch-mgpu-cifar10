# -*- coding: utf-8 -*-
'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function
from sqlite3 import NotSupportedError

import torch
import torch.nn as nn

import logging
import copy

from models import *
from utils import *
from data_loader import *
import pdb
from plot_results import *
from Fed import FedAvg

from common import *

logger = logging.getLogger('__name__')
logger.setLevel('INFO')


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
    parser.add_argument('--aggregation_mode', default='distillation')

    ######################### Distillation Parameters #########################
    parser.add_argument('--public_distill', action='store_true', help="use public data to distill")
    parser.add_argument('--exist_loader', action='store_true', help="there is exist loader")
    parser.add_argument('--save_loader', action='store_true', help="save trainloaders")
    parser.add_argument('--alpha', default=100, type=float, help='alpha for dirichle partition')
    parser.add_argument('--lamb', default=0.5, type=float, help='lambda for distillation')
    parser.add_argument('--temp', default=1, type=float, help='temperature for distillation')
    parser.add_argument('--public_percent', default=0.5, type=float, help='percentage training data to be public')
    parser.add_argument('--distill_percent', default=1, type=float, help='percentage of public data use for distillation')
    parser.add_argument('--vary_epoch', action='store_true', help="change the number of local epochs of edges")
    parser.add_argument('--save_confidence', action='store_true', help="save all the norm values for debug purpose")

    ######################### Aggregation Parameters #########################
    parser.add_argument('--selection', action='store_true', help="enable selection method")
    parser.add_argument('--emb_mode', default='dlc', help="[dlc, wavg]")
    parser.add_argument('--num_drop', default=1, type=int, help='number of edges to be dropped')
    parser.add_argument('--finetune', action='store_true', help="finetune the cloud model") # test fine-tune
    parser.add_argument('--finetune_epoch', default=10, type=int, help='number of epochs for finetune')
    parser.add_argument('--finetune_percent', default=0.2, type=float, help='percentage of data to finetune')
    parser.add_argument('--sample_percent', default=1, type=float, help='percentage of private data in each worker')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_arguments()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
                # The first worker has classes [0,1], the second worker has classes [2,3]
                # Edge data is non-overlapped
                if args.partition_mode == 'uniform':
                    trainloaders = get_subclasses_loaders(trainset_private, args.num_workers, client_classes, num_workers=4, non_overlapped=True, seed=100)
                    class_select = args.num_workers * client_classes

                if args.partition_mode == 'dirichlet':
                    
                    # s = args.split if args.split == 1 else args.split * args.num_workers
                    extract_trainset = extract_classes(trainset_private, args.split, dataset=args.dataset, workerid=0)
                    # use 1 thread worker instead of 4 in the single gpu case
                    trainloaders = get_dirichlet_loaders(extract_trainset, n_clients=args.num_workers, alpha=args.alpha, num_workers=1, seed=100)
                    class_select = client_classes

                logger.info(f"Cloud Test data")
                testloader_cloud = get_subclasses_loaders(testset, n_clients=1, client_classes=class_select, num_workers=4, non_overlapped=True, seed=100)

                logger.info(f"Edge Test data")
                
                if args.partition_mode == 'uniform':
                    testloaders = get_subclasses_loaders(testset, args.num_workers, client_classes, num_workers=4, non_overlapped=True, seed=100)

                elif args.partition_mode == 'dirichlet':
                    testloaders = get_subclasses_loaders(testset, args.num_workers, client_classes, num_workers=4, non_overlapped=False, seed=100)
                
                else:
                    raise NotImplementedError("Not supported partition mode")

                logger.info(f"Cloud Train loader")
                trainloader_cloud = get_subclasses_loaders(trainset_public, n_clients=1, client_classes=client_classes, num_workers=4, non_overlapped=True, seed=100)

            # Use private data to perform distillation       
            else:
                trainloaders = get_subclasses_loaders(trainset, args.num_workers, client_classes, num_workers=4, non_overlapped=True, seed=100)
                trainloader_cloud = get_subclasses_loaders(trainset, n_clients=1, client_classes=int(args.num_workers * client_classes), num_workers=4, non_overlapped=True, seed=100)
                testloader_cloud = get_subclasses_loaders(testset, n_clients=1, client_classes=int(args.num_workers * client_classes), num_workers=4, non_overlapped=True, seed=100)
                testloaders = get_subclasses_loaders(testset, args.num_workers, client_classes, num_workers=4, non_overlapped=True, seed=100)

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

    ################### Define models based on aggregation mode ###################

    for i in range(args.num_workers):
            net = build_model_from_name(args.net, num_classes, device)
            nets.append(net)

    if args.aggregation_mode == 'distillation':
        cloud_net = build_model_from_name(args.cloud, num_classes, device)

    elif args.aggregation_mode == 'fedavg':
        cloud_net = build_model_from_name(args.net, num_classes, device)
            
    else:
        raise NotSupportedError("Aggregation mode not supported. Use 'distillation' or 'fedavg")

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
                    if args.aggregation_mode == 'fedavg':
                        
                        logger.info("Copying cloud weights")
                        cloud_w = cloud_net.state_dict()
                        
                        # edge model loads the cloud weights
                        logger.info("Edge loading cloud weights")
                        for i, model in enumerate(nets):
                            model.load_state_dict(cloud_w)
                    
                    logger.info(f"Training the {i} edge")
                    run_train(nets[i], round, args, trainloaders[i], testloader_cloud, testloaders[i], i, device, False, 'local', 'edge_' + str(i))

        if args.aggregation_mode == 'distillation': 

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
        
        elif args.aggregation_mode == 'fedavg':
            
            all_weights = []

            for i, model in enumerate(nets):

                w = model.state_dict()
                # edge_models.append(model)              
                all_weights.append(copy.deepcopy(w))
        
            logger.debug(f"Length of all_weights: {len(all_weights)}")

            averaged_weights = FedAvg(all_weights)
            cloud_net.load_state_dict(averaged_weights)
            
            ########### Calculate accuracy ###########            
            criterion_edge = nn.CrossEntropyLoss()
            acc, best_acc = test(round, args, cloud_net, criterion_edge, testloader_cloud, device, 'fedavg')
            logger.info(f"The fedavg accuracy is: {acc}")
            csv_name = 'acc_fedavg'  + '.csv'
            path = 'results/' + args.workspace
            write_csv(path, csv_name, str(acc))