from train_cifar import build_model_from_name, print_total_params
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models import *
import torchvision.transforms as transforms
import torchvision
import logging
from utils import write_csv, get_time, progress_bar
import sys
import time
import math
import torch.nn.init as init
import argparse
import os
from data_loader import extract_classes, split_train_data, get_dirichlet_loaders
import time
from torchvision.datasets import ImageFolder
from torchvision import datasets, models, transforms


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
fmt_str = '%(name)s - %(levelname)s - %(message)s'
fmt_file = '%(asctime)s - %(name)s [%(levelname)s]: %(message)s'

c_handler = logging.StreamHandler()
logger.addHandler(c_handler)
c_format = logging.Formatter(fmt_str)
c_handler.setFormatter(c_format)

epoch=200
batch_size=128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch=0

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net', default='res8')
parser.add_argument('--opt', default='sgd')
parser.add_argument('--workspace', default='test_workspace')
parser.add_argument("--percent_classes", default=1, type=float, help="how many classes to classify")
parser.add_argument("--percent_data", default=1, type=float, help="percentage of data to use for training")
parser.add_argument('--dataset', default='imagenet', help="use imagenet or tiny-imagenet")
args = parser.parse_args()


def get_iterator_imagenet(imagenetpath, batch_size, nthread=4, mode='train'):
    # refer attention transfer and open_lth
    imagenetpath = os.path.expanduser(imagenetpath)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    print("| setting up data loader...")
    if mode == "train":

        if "tiny-imagenet-200" in imagenetpath:
            traindir = os.path.join(imagenetpath, 'train')
            ds = ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(64), # refer open_lth
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
        else:
            traindir = os.path.join(imagenetpath, 'train')
            ds = ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(0.8, 1.25)), # refer open_lth
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:

        if "tiny-imagenet-200" in imagenetpath:
            valdir = os.path.join(imagenetpath, 'val')
            ds = ImageFolder(valdir, transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
        ]))

        else:
            valdir = os.path.join(imagenetpath, 'val')
            ds = ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    size = len(ds)
    # total_classes = int(max(ds.targets) + 1)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=mode, num_workers=nthread, pin_memory=True), size


def train(epoch):
    print('\nEpoch: %d' % epoch)
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
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    return train_loss/(batch_idx+1)

def save_checkpoint(net, acc):
    state = {
        'net': net.state_dict(),
        'acc': acc,
    }
    checkpoint_dir = 'results/' + args.workspace + '/checkpoint/'
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, checkpoint_dir + 'checkpoint.pt')

def test_acc(epoch):
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
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc = 100.*correct/total
    
    return acc

def check_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)    

t0 = time.time()
# print('==> Preparing data..')
if args.dataset == "imagenet":
    print('==> Preparing ImageNet..')
    imagenet_path = '/home/users/kzhao27/imagenet_data'
else:
    print('==> Preparing Tiny-ImageNet..')
    imagenet_path = '/home/users/kzhao27/tiny-imagenet-200'

trainloader, size = get_iterator_imagenet(imagenet_path, 128, mode='train')
testloader, _ = get_iterator_imagenet(imagenet_path, 128, mode='val')

print(trainloader)

print(len(trainloader), size)
# net = resnet18(num_classes=1000)
net = resnet10_pytorch(pretrained=False)
# net = models.resnet18(pretrained=False)

# transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
#     ])

# transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
#     ])

# trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
# testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
# length = len(trainset)

# if args.percent_classes != 1:
#     trainset = extract_classes(trainset, args.percent_classes, workerid=0)
#     testset = extract_classes(testset, args.percent_classes, workerid=0)
    
#     if args.percent_data != 1:
#         trainloaders = get_dirichlet_loaders(trainset, n_clients=int(1/args.percent_data), alpha=100)
#         trainloader = trainloaders[0]
#         # trainset, part_b = split_train_data(trainset, args.percent_data)
#         # target_length = int(length * args.percent_classes * args.percent_data)
#         # assert len(trainset) == target_length, f"Wrong target length. trainset: {len(trainset)} target: {target_length}"

#     else:
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# print('==> Building model..')

# net = resnet6_imagenet(num_classes=100)

# if args.net == 'res4':
#     net = resnet4(num_classes=100)
# elif args.net == 'res8':
#     net = resnet8(num_classes=100)
# elif args.net == 'res6':
#     net = resnet6(num_classes=100)
# elif args.net =='res18':
#     net = resnet18(num_classes=100)
# elif args.net =='res34':
#     net = resnet34(num_classes=100)

# else:
#     logger.debug("Not supported model")

print_total_params(net)
net = net.to(device)
print(device)


if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()

if args.opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr)

elif args.opt =='adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

t1 = time.time()
print(f"Data time: {t1 - t0} seconds")
t2 = time.time()

strtime = get_time()
root = 'results/' + args.workspace
check_dir(root)
# print(net)

best_acc = 0
for epoch in range(start_epoch, start_epoch + epoch):
    trainloss = train(epoch)
    acc = test_acc(epoch)
    logger.debug(f"The result is: {acc}")
    # write_csv('acc_' + args.workspace + '_worker_0' + 'res8_' + '.csv', str(acc))
    write_csv('results/' + args.workspace, 'acc_' +  str(args.net) + '_' + strtime + '.csv', str(acc))
    if acc > best_acc:
        logger.debug(f"Saving model...")
        save_checkpoint(net, acc)
        best_acc = acc
logger.debug(f"The best acc is: {acc}")
t3 = time.time()
print(f"Training time: {t3 - t2} seconds")
