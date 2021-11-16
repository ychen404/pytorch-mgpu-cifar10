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
import torch.utils.data.sampler as Sampler

class YourSampler(Sampler):
    # def __init__(self, mask):
    #     self.mask = mask

    # def __iter__(self):
    #     return (self.indices[i] for i in torch.nonzero(self.mask))

    # def __len__(self):
    #     return len(self.mask)
    pass

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

even = list(range(0, len(trainset), 2))
odds = list(range(1, len(trainset), 2))

print(even)

sampler1 = YourSampler(1)
sampler2 = YourSampler(2)

trainloader_sampler1 = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          sampler = sampler1, shuffle=False, num_workers=2)
trainloader_sampler2 = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          sampler = sampler2, shuffle=False, num_workers=2)

print(len(trainloader_sampler1), len(trainloader_sampler2))