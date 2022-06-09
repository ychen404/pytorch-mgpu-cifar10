import json
import csv
import numpy as np

path = 'results/check_confidence_100_5workers/mock_confidence.json'
# path = '/home/users/yitao/Code/pytorch-mgpu-cifar10/results/check_confidence_0.01_5workers/confidence.json'

with open(path, 'r') as f:
    data = json.load(f)

# for d in
data_np = np.asarray(data)
# print(type(data_np))
# print(np.var(data_np, axis=1))

for d in data_np:
    print(np.var(d[1:]))
#     print(np.var(d[:1]))