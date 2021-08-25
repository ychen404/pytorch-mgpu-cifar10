import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pdb

def get_cifar(num_classes=100, dataset_dir="./data", batch_size=128):

    if num_classes == 10:
        print("Loading CIFAR10...")
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        print("Loading CIFAR100...")
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
    ])

    train_data = dataset(root=dataset_dir, train=True,
                       download=True, transform=train_transform)

    test_data = dataset(root=dataset_dir, train=False,
                          download=True,
                          transform=test_transform)

    return train_data, test_data

def extract_classes(train_data, split, workerid=0):

    # we can change this to go through the data once and return all the classes for all the workers
    classes = []
    total_classes = int(max(train_data.targets) + 1)
    num_classes = int(total_classes * split)
    
    upper = int(max(train_data.targets) * split)
    # print(f"Upper: {upper}")
    start = workerid * num_classes + 0
    end = workerid * num_classes + upper
    print(f"Worker: {workerid}; start: {start}; end: {end}; num_classes: {num_classes}")

    for data in train_data:
        if start <= data[1] <= end:
          classes.append(data)
    
    return classes


cifar100_train_data, _ = get_cifar(num_classes=100, batch_size=128)
extract_trainset = extract_classes(cifar100_train_data, 0.3)
train_loader = torch.utils.data.DataLoader(extract_trainset,
                                          batch_size=128,
                                          num_workers=4,
                                          pin_memory=True, shuffle=False)

counter = {}
for idx, (images, targets) in enumerate(train_loader):
    if idx == 0:
        alpha, beta, gamma = 0, 0, 0
        print(f"batch size = {targets.shape[0]}")
        for target in targets:
        # counter[target.tolist()] = counter.get(target.tolist(), 0) + 1
            if 0 <= target.item() <= 9:
                alpha += 1
            elif 10 <= target.item() <= 19:
                beta += 1
            else:
                gamma += 1
    else:
        break

print(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}")

sum_v = 0

for k, v in counter.items():
    sum_v += v
print(f"total: {sum_v}")
alpha /= 128
beta /= 128
gamma /= 128

print(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}")

