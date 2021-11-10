import numpy as np
import torch
import torch.nn as nn
from numpy import dot
from numpy.linalg import norm
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

def cosine_similarity(a, b):
    
    """
    Use the channel as the vector to calculate the cosine similarity
    Assume the input format is (batch, num_channels, w, h)
    Summing up all the values as output
    """
    output = F.cosine_similarity(a.view(1, 2, -1), b.view(1, 2, -1), dim=2)
    output = torch.sum(output)

    return output

def cosineSimilarity(x1, x2):
    x1_sqrt = torch.sqrt(torch.sum(x1 ** 2))
    x2_sqrt = torch.sqrt(torch.sum(x2 ** 2))
    return torch.div(torch.sum(x1 * x2), max(x1_sqrt * x2_sqrt, 1e-8))


def calculate_similarity_index(public_loader, private_loader, device):
    
    result = []
    similarity_index = torch.zeros(1)
    similarity_index = similarity_index.to(device)
    counter = 0
    for batch_idx, (inputs, targets) in enumerate(private_loader):        
        # print(f"batch: {batch_idx}")
        inputs = inputs.to(device)
        for pub_batch_idx, (inputs_pub, targets_pub) in enumerate(public_loader):
            print(f"Counter: {counter}")
            counter += 1
            inputs_pub = inputs_pub.to(device)
            # print("public,", inputs_pub.shape, pub_batch_idx, "private", inputs.shape, batch_idx)
            if inputs_pub.shape != inputs.shape:
                continue 
            similarity_index += cosine_similarity(inputs_pub, inputs)
            similarity_index /= len(public_loader)
        result.append(similarity_index.item())
    return result

if __name__ == "__main__":
    
    
    """
    https://discuss.pytorch.org/t/underrstanding-cosine-similarity-function-in-pytorch/29865/10
    
    ptrblck explained the pytorch cosine similarity

    The below example uses the 10x10 pixels as the vector to calculate the cosine similarity
    Each channel would therefore hold a 100-dimensional vector pointing somewhere 
    and you could calculate the similarity between the channels.
    """

    # calculate the similarity along the channel 
    # a = torch.randn(1, 3, 10, 10)
    # b = torch.randn(1, 3, 10, 10)
    # # output = F.cosine_similarity(a.view(1, 2, -1), b.view(1, 2, -1), 2)
    # zeros = torch.zeros(1)
    # print(zeros)
    # output = cosine_similarity(a, b)
    # zeros += output
    # print(output)
    # print(zeros.item())

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)


    set_1 = list(range(0, 1280, 1))
    set_2 = list(range(1280,2560, 1))
    # print(set_1, set_2)

    trainset_1 = torch.utils.data.Subset(trainset, set_1)
    trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=128, shuffle=True, num_workers=4)
    
    trainset_2 = torch.utils.data.Subset(trainset, set_2)
    trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=128, shuffle=True, num_workers=4)
    

    a = torch.randn(1, 3, 10, 10)
    b = torch.randn(1, 3, 10, 10)


    si = calculate_similarity_index(trainloader_1, trainloader_2, 'cuda')
    print(si)