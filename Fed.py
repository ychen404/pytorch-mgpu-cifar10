import copy
import torch
from torch import nn
from math import pi
from math import cos
from math import floor

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


if __name__ == "__main__":
    n_epochs = 100
    n_cycles = 5
    lrate_max = 0.001
    
    # series = [cosine_annealing(i, n_epochs, n_cycles, lrate_max) for i in range(n_epochs)]
    # print(series)