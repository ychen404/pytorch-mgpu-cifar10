from utils import *
from train_cifar import *
import pdb

device = 'cuda:0'
name = 'res18'
num_classes = 100
public_percent = 0.5
finetune_percent = 1
num_workers = 5
cls_per_worker = 2
start_epoch = 0
epoch = 50
workspace = 'iid_5_workers_res6_2_cls_public_distill/'
batch_size = 128
split = 0.1


def get_trainloader(train_data, batch_size):
    """Get a single train loader given train data"""
    train_loader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=4, 
                                                pin_memory=True)
    return train_loader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = build_model_from_name(name, num_classes, device)

net = load_edge_checkpoint_fullpath(net, 'results/iid_5_workers_res6_2_cls_public_distill/checkpoint/cloud_ckpt.t7')
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

print(net)
trainset, testset = get_cifar100()
trainset_public, trainset_private = split_train_data(trainset, public_percent)
trainset_public, _ = split_train_data(trainset_public, finetune_percent)

# pdb.set_trace()
# trainloader = get_subclasses_loaders(trainset_public, n_clients=1, client_classes= num_workers * cls_per_worker, num_workers=4, seed=100)
trainloader = get_trainloader(trainset_public, batch_size)
print(len(trainloader))
# exit()
criterion = nn.CrossEntropyLoss()

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

def test_acc(testloader):
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

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

_, testloader_iid = get_worker_data_hardcode(trainset, split, workerid=0)

# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)


for epoch in range(start_epoch, start_epoch + epoch):
    trainloss = train(epoch)
    acc = test_acc(testloader_iid)
    print(acc)
    logger.debug(f"The result is: {acc}")
    # write_csv('acc_' + args.workspace + '_worker_0' + 'res8_' + '.csv', str(acc))
    write_csv('results/' + workspace, 'acc_' +  str(name) + '_' + 'finetune' + '.csv', str(acc))