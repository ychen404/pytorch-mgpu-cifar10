import logging
import torch
from utils import progress_bar
import os

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from utils import get_time, write_csv, freeze_net
import argparse
import pdb

logger = logging.getLogger('__name__')

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
    
    logger.debug("===> BEST ACC. PERFORMANCE: %.2f%%" % (best_acc))


def distill(
    epoch,  
    edge_nets, 
    cloud_net, 
    optimizer, 
    distill_loader, 
    device,
    num_workers,
    split,
    distill_percent, 
    dataset,
    average_method,
    select_mode,
    emb_mode,
    num_drop,
    selection=False,
    lambda_=1,
    temperature_=1,
    save_confidence=False,
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
    for batch_idx, (inputs, targets) in enumerate(distill_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        
        if batch_idx % 10 == 0:
            logger.debug(f"Processing batch {batch_idx}")
        
        optimizer.zero_grad()
        counter += 1
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
        
        # Other methods
        else:
            # Use torch.zeros(128,100) to create the placeholder
            out_t = torch.zeros((size_curr_batch, total_classes), device=device)
            t_max = torch.zeros((size_curr_batch, total_classes), device=device)
            
            save_all_output = [[] for _ in range(len(inputs))]
            confidence_results = [[] for _ in range(len(inputs + 1))] # Add one position to store the true label

            # emb_mode determines how do we use emb
            # 'wavg' uses the norm of the emb to perform weighted average on softmax
            # 'dlc' uses the norm of the emb to drop edge workers 
            if emb_mode == 'wavg' or emb_mode == 'fedet': # if not dropping any edge, the holder should be able to hold all the edge res
                temp_res = torch.empty((len(edge_nets), total_classes), device=device)    
            
            else:
                # The output size of each sample is torch.Size([1, 10]) for cifar10
                num_left_edge = len(edge_nets) - num_drop
                temp_res = torch.empty((num_left_edge, total_classes), device=device)

            emb_flag = False
            
            # TODO: Need to add a method to distinguish with other
            if emb_mode == 'dlc' or emb_mode == 'wavg' or emb_mode == 'fedet':
                emb_flag = True
                for idx, (input, target) in enumerate(zip(inputs, targets)):
                    input = input.unsqueeze(0)
                    
                    # Save the true label to the confidence output
                    if save_confidence:
                                confidence_results[idx].append(target.item())
                    
                    for i, edge_net in enumerate(edge_nets):
                        # TODO: use the emb model as only case to simplify
                        if hasattr(edge_net, 'emb') and edge_net.emb:
                        
                            edge_out, emb = edge_net(input)
                            embDim = edge_net.get_embedding_dim()
                            emb = emb.data.cpu().numpy()

                            if emb_mode == 'fedet':
                                # Save the softmax output from each image
                                # pdb.set_trace()
                                
                                edge_softmax = F.softmax(edge_out, dim=1)
                                
                                softmax_var = torch.var(edge_softmax)
                                save_all_output[idx].append((softmax_var.item(), edge_softmax))
                                # pdb.set_trace()
                                
                            if emb_mode == 'dlc' or emb_mode == 'wavg':
                     
                                emb_norm = return_edge_norm(edge_out, emb)
                                save_all_output[idx].append((emb_norm, edge_out))

                                if save_confidence:
                                    confidence_results[idx].append(emb_norm)

                                #edge_predicted = edge_out.max(1)
                                # No need to check this. 
                                # This is already done before. 
                                """
                                edge_predicted is a tensor consists of the value and the index:
                                torch.return_types.max(
                                values=tensor([6.3136], device='cuda:0'),
                                indices=tensor([8], device='cuda:0'))
                                
                                target is a tensor with the index: tensor(8, device='cuda:0')
                                
                                In order to compare them, we need to index twice on the edge_predicted
                                edge_predicted[1] -> (tensor([8], device='cuda:0')
                                edge_predicted[1][0] -> (tensor(8, device='cuda:0'))
                                """
                                            
                    sorted_output = save_all_output[idx] # save the outputs from a batch 

                    if emb_mode == 'wavg': # weighted average of all edge
                        sum_norm = sum(e[0] for e in sorted_output)
                        for i, e in enumerate(sorted_output):
                            # Low norm value has a higher weight
                            temp_res[i] = ((sum_norm - e[0]) / sum_norm) * e[1]

                    elif emb_mode == 'dlc': # dropping some edge workers
                        sorted_output.sort(key = lambda x : x[0]) # use the emb_norm to sort
                        
                        # when num_drop == 0, this case is the same as FedDF
                        if num_drop == 0:
                            for i, e in enumerate(sorted_output):
                                temp_res[i] = e[1] # the 1st idx in e is the model output
                        else:
                            for i, e in enumerate(sorted_output[:-num_drop]):
                                # Large norm means low confidence
                                # Drop the last worker with the largest norm
                                temp_res[i] = e[1] # the 1st idx in e is the model output
                    
                    elif emb_mode == 'fedet':
                        # pdb.set_trace()
                        # Higher the variance, higher confidence
                        
                        sum_var = sum(e[0] for e in sorted_output)
                        for i, e in enumerate(sorted_output):
                            temp_res[i] = (e[0] / sum_var) * e[1]

                    else:
                        raise NotImplementedError("Not supported emb mode")
                        
                    # Calculate the average of the output of each sample
                    # Each output is torch.size([1,10])
                    # Use dim=0, the size of the mean is torch.size([10]), use keepdim to maintain the torch.size([1, 10])
                    out_t_temp.append(temp_res.mean(dim=0, keepdim=True))                
            
                out_t = torch.cat(out_t_temp, dim=0) # use dim 0 to stack the tensors
            
            
            #### Calculate model outputs from all edge models without dropping
            else:
                for i, edge_net in enumerate(edge_nets):
                    edge_net = edge_net.to(device)
                    out_t += edge_net(inputs)
            
            # Except fedet, other modes calculate softmax at the last step
            if emb_mode != 'fedet':
                t_max += F.softmax(out_t / T, dim=1)

            else: # fedet t_max is already available
                t_max = out_t
            
            if average_method == 'equal':
                logger.debug("Equal weights")
                t_max = t_max / num_workers
            
            elif average_method == 'grad':
                # Use gradient magnitude to select workers
                # So nothing needs to be done here for now.
                # We may need to add some additional operations 
                pass

            else: 
                raise NotImplementedError("Not support weighted average now")

        loss_kd = kd_fun(s_max, t_max) / size_curr_batch
        loss = loss_fun(out_s, targets)
        loss_kd =(1 - lambda_) * loss + lambda_ * T * T * loss_kd

        loss_kd.backward()
        optimizer.step()
        train_loss += loss_kd.item()
        
        # We do not use the true labels to distill
        # But we can still use them to check the training accuracy for debug purpose
        # The bar is not accumulating the images
        # _, predicted = out_s.max(1)
        # total += targets.size(0)
        # correct += torch.sum(predicted == targets).float()

        # progress_bar(batch_idx, len(distill_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # for only the progress_bar
        progress_bar(batch_idx, len(distill_loader))
        
    
    if save_confidence:
        return train_loss/(batch_idx+1), confidence_results
    else:
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
        res = distill(epoch,
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
                    emb_mode=args.emb_mode,
                    num_drop=args.num_drop,
                    selection=selection, 
                    lambda_=args.lamb,
                    temperature_=args.temp,
                    save_confidence=args.save_confidence)

        acc, best_acc = test(epoch, args, cloud, criterion_cloud, testloader_cloud, device, 'cloud')

        logger.debug(f"The result is: {acc}")        
        write_csv('results/' + args.workspace, prefix + strtime + '.csv', str(acc))
        
        # Debug confidence
        # Now the distill function returns [trainloss, confidence_results]
        if args.save_confidence:
            trainloss = res[0]
            confidence_results = res[1]
            # write_csv('results/' + args.workspace, 'confidence.csv', str(confidence_results))
            from utils import save_json
            save_json('results/' + args.workspace, 'confidence.json', confidence_results)

        else:
            trainloss = res

        list_loss.append(trainloss)

    logger.debug("===> BEST ACC. DISTILLATION: %.2f%%" % (best_acc))

    return cloud