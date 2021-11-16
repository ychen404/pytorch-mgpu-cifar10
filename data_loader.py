from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import pdb

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
fmt_str = '%(name)s - %(levelname)s - %(message)s'
fmt_file = '%(asctime)s - %(name)s [%(levelname)s]: %(message)s'

c_handler = logging.StreamHandler()
logger.addHandler(c_handler)
c_format = logging.Formatter(fmt_str)
c_handler.setFormatter(c_format)


def get_cifar10_loader(args):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def get_cifar100():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_cifar100_transfromtest():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    return transform_test

def split_train_data(train_data, split=0.5):
    
    length = len(train_data)
    
    public_part = int(split * length)
    private_part = length - public_part

    print(f"Length: {length}; public_part: {public_part}, private_part: {private_part} ")

    public, private = torch.utils.data.random_split(train_data, [public_part, private_part])

    return public, private

def extract_classes(train_data, split, workerid=0):

    # we can change this to go through the data once and return all the classes for all the workers
    classes = []
    # total_classes = int(max(train_data.targets) + 1)
    total_classes = 100    
    num_classes = int(total_classes * split)
    # upper = int(max(train_data.targets) * split)
    upper = int(total_classes * split)
    # print(f"Upper: {upper}")
    start = workerid * num_classes + 0
    end = workerid * num_classes + upper
    
    logger.debug(f"Worker: {workerid}; start: {start}; end: {end}; num_classes: {num_classes}")

    # Pay attention to the bounds
    for data in train_data:
        if start <= data[1] < end:
          classes.append(data)
    
    return classes


def get_worker_data(trainset, args, workerid)->torch.utils.data.dataloader.DataLoader:

    """Get train data for each edge worker"""
    transform_test = get_cifar100_transfromtest()

    logger.debug(f"Extracting training data")
    extract_trainset = extract_classes(trainset, args.split, workerid)
    trainloader = torch.utils.data.DataLoader(extract_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    logger.debug(f"Extracting test data")
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    extract_testset = extract_classes(testset, args.split, workerid)
    testloader = torch.utils.data.DataLoader(extract_testset, batch_size=128, shuffle=False, num_workers=4)

    return trainloader, testloader

def get_worker_data_hardcode(
    trainset, 
    split, 
    workerid, 
    disjoint=False
    )->torch.utils.data.dataloader.DataLoader:

    """Get disjoint train data for testing accuracy including the third worker's data"""

    transform_test = get_cifar100_transfromtest()

    # added hardcoded for a simple test
    if disjoint:
        logger.debug("Disjoint training data")
        extract_trainset_1 = extract_classes(trainset, 0.1, 0)
        extract_trainset_2 = extract_classes(trainset, 0.1, 2)
        extract_trainset = extract_trainset_1 + extract_trainset_2
    else:
        extract_trainset = extract_classes(trainset, split, workerid)

    trainloader = torch.utils.data.DataLoader(extract_trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    if disjoint:
        logger.debug("Disjoint testing data")

        extract_testset_1 = extract_classes(testset, 0.1, 0)
        extract_testset_2 = extract_classes(testset, 0.1, 2)
        extract_testset = extract_trainset_1 + extract_trainset_2
    else:
        extract_testset = extract_classes(testset, split, workerid)
        logger.debug(f"Length testset: {len(testset)}")

    logger.debug(f"Length extract_testset: {len(extract_testset)}")
    testloader = torch.utils.data.DataLoader(extract_testset, batch_size=128, shuffle=False, num_workers=4)

    return trainloader, testloader

def get_loader(train_data, args):
    """Get a single train loader given train data"""
    train_loader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                num_workers=4, 
                                                pin_memory=True)
    return train_loader


def extract_targets(train_data):
  
  targets = []
  for data in train_data:
    targets.append(data[1])

  return targets

def split_dirichlet(labels, n_clients, n_data, alpha, double_stochstic=True, seed=0):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    np.random.seed(seed)

    if isinstance(labels, torch.Tensor):
      labels = labels.numpy()

    n_classes = np.max(labels)+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)
  
    return client_idcs


def split_uniform(labels, n_clients, client_classes, seed=0):
    '''Splits data among the clients according to a uniformlly, i.e., each client has a few classes
        This numpy based implementation is much faster than my previous one
    '''

    np.random.seed(seed)

    if isinstance(labels, torch.Tensor):
      labels = labels.numpy()

    n_classes = np.max(labels)+1
    class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
           for y in range(n_classes)]

    # print(len(class_idcs))
    # print(len(class_idcs[0]))
    client_idcs = [[] for _ in range(n_clients)]

    # Get the idcs for each client data
    # for example, client 0 has the first two class, so the idxs are 2x0 and 2x0 + 1
    # 2 is client_classes, 0 and 1 are cc
    for idx, c in enumerate(class_idcs):
        for i in range(n_clients):
            for cc in range(client_classes):
                if idx == client_classes * i + cc:
                    client_idcs[i] += [c]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)
  
    return client_idcs

def print_split(idcs, labels):
    n_labels = np.max(labels) + 1 
    print("Data split:")
    
    splits = []
    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
        splits += [split]
        if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
            print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
        elif i==len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

    print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
    print()


def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    return x


def get_dirichlet_loaders(train_data, n_clients=3, alpha=0, batch_size=128, n_data=None, num_workers=4, seed=0):

    # Check if it is train_data object
    if not isinstance(train_data, torchvision.datasets.cifar.CIFAR100):
        train_data_targets = extract_targets(train_data)

    else:
        train_data_targets = train_data.targets
    
    subset_idcs = split_dirichlet(train_data_targets, n_clients, n_data, alpha, seed=seed)
    client_data = [torch.utils.data.Subset(train_data, subset_idcs[i]) for i in range(n_clients)]

    client_loaders = [torch.utils.data.DataLoader(subset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=num_workers, 
                                                    pin_memory=True) for subset in client_data]


    return client_loaders

def get_subclasses_loaders(train_data, n_clients=3, client_classes=2, batch_size=128, num_workers=4, seed=0):

    # Check if it is train_data object
    if not isinstance(train_data, torchvision.datasets.cifar.CIFAR100):
        train_data_targets = extract_targets(train_data)

    else:
        train_data_targets = train_data.targets

    subset_idcs = split_uniform(train_data_targets, n_clients, client_classes, seed=seed)
    client_data = [torch.utils.data.Subset(train_data, subset_idcs[i]) for i in range(n_clients)]

    client_loaders = [torch.utils.data.DataLoader(subset, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=num_workers, 
                                                pin_memory=True) for subset in client_data]
    return client_loaders