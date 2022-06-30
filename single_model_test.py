from xml.dom import NotSupportedErr
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
import time
import argparse
import os
from data_loader import extract_classes, split_train_data, get_dirichlet_loaders
import time
from imagenetLoad import ImageNetDownSample
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
fmt_str = '%(name)s - %(levelname)s - %(message)s'
fmt_file = '%(asctime)s - %(name)s [%(levelname)s]: %(message)s'

c_handler = logging.StreamHandler()
logger.addHandler(c_handler)
c_format = logging.Formatter(fmt_str)
c_handler.setFormatter(c_format)

# epoch=200
# batch_size=128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch=0

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
parser.add_argument('--net', default='res8')
parser.add_argument('--opt', default='sgd')
parser.add_argument('--lr_sched', default=None, help='multistep, cos')
parser.add_argument('--workspace', default='test_workspace')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument("--percent_classes", default=1, type=float, help="how many classes to classify")
parser.add_argument("--percent_data", default=1, type=float, help="percentage of data to use for training")

args = parser.parse_args()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    edge_net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.lr_sched == 'cos':
            lr_scheduler.step()
            print(f"current lr: {lr_scheduler.get_lr()}")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = edge_net(inputs)
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
    edge_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = edge_net(inputs)
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

if args.dataset == "cifar100":
    print('==> Preparing CIFAR-100 data..')
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
    length = len(trainset)


elif args.dataset == "cifar10":
    print('==> Preparing CIFAR-10 data..')
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
    
    if args.percent_data != 1:
        trainset, part_b = split_train_data(trainset, args.percent_data)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


elif args.dataset == "imagenet":
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    datapath = '/home/users/yitao/downsampled_imagenet/imagenet32'

    trainset = ImageNetDownSample(root=datapath, train=True, transform=transform_train)
    testset = ImageNetDownSample(root=datapath, train=False, transform=transform_test)

if args.percent_classes != 1:
    trainset = extract_classes(trainset, args.percent_classes, workerid=0)
    testset = extract_classes(testset, args.percent_classes, workerid=0)
    
    if args.percent_data != 1:
        trainloaders = get_dirichlet_loaders(trainset, n_clients=int(1/args.percent_data), alpha=100)
        trainloader = trainloaders[0]

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

print('==> Building model..')

nc = 100 if args.dataset == "cifar100" else 1000

if args.dataset == 'cifar10':
    nc = 10
elif args.dataset == 'cifar100':
    nc = 100
else:
    nc = 1000

if args.net == 'res4':
    edge_net = resnet4(num_classes=nc)
elif args.net == 'res8':
    edge_net = resnet8(num_classes=nc)
elif args.net == 'res8_aka':
    edge_net = resnet8_aka(num_classes=nc)
elif args.net == 'res6':
    edge_net = resnet6(num_classes=nc)
elif args.net =='res18':
    edge_net = resnet18(num_classes=nc)
elif args.net =='res34':
    edge_net = resnet34(num_classes=nc)
elif args.net =='vgg19':
    edge_net = VGG('VGG19')
else:
    raise NotSupportedErr("Not supported model")

print_total_params(edge_net)
edge_net = edge_net.to(device)
if device == 'cuda':
    edge_net = torch.nn.DataParallel(edge_net) # make parallel
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if args.opt == 'sgd':
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(edge_net.parameters(), lr=args.lr)

elif args.opt =='adam':
    optimizer = optim.Adam(edge_net.parameters(), lr=args.lr, weight_decay=5e-4)

if args.lr_sched == 'multistep':
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150])
if args.lr_sched == None:
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[65533])
elif args.lr_sched == 'cos':
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0)
    # exit()
t1 = time.time()
print(f"Data time: {t1 - t0} seconds")
t2 = time.time()

strtime = get_time()
root = 'results/' + args.workspace
check_dir(root)

writer = SummaryWriter(log_dir=root)

best_acc = 0
for epoch in range(args.epoch):
    print('current epoch {}, current lr {:.5e}'.format(epoch, optimizer.param_groups[0]['lr']))
    trainloss = train(epoch)
    writer.add_scalar('Loss/train', trainloss, epoch)
    # writer.add_scalar('Loss/test', np.random.random(), n_iter)
    # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)

    
    if args.lr_sched == 'multistep':
        lr_scheduler.step()
    acc = test_acc(epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)

    logger.debug(f"The result is: {acc}")
    write_csv('results/' + args.workspace, 'acc_' +  str(args.net) + '_' + strtime + '.csv', str(acc))
    if acc > best_acc:
        logger.debug(f"Saving model...")
        save_checkpoint(edge_net, acc)
        best_acc = acc
logger.debug(f"The best acc is: {acc}")
t3 = time.time()
print(f"Training time: {t3 - t2} seconds")