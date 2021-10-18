import numpy as np
import torch
import torch.nn as nn
from numpy import dot
from numpy.linalg import norm
import torch.nn.functional as F

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

if __name__ == "__main__":
    
    
    """
    https://discuss.pytorch.org/t/underrstanding-cosine-similarity-function-in-pytorch/29865/10
    
    ptrblck explained the pytorch cosine similarity

    The below example uses the 10x10 pixels as the vector to calculate the cosine similarity
    Each channel would therefore hold a 100-dimensional vector pointing somewhere 
    and you could calculate the similarity between the channels.
    """

    # calculate the similarity along the channel 
    a = torch.randn(1, 3, 10, 10)
    b = torch.randn(1, 3, 10, 10)
    output = F.cosine_similarity(a.view(1, 2, -1), b.view(1, 2, -1), 2)
    print(output)