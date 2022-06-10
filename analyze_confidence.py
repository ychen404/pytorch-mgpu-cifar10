import json
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

sns.set_theme()
# path = 'results/check_confidence_100_5workers/mock_confidence.json'
# path = 'results/check_confidence_100_5workers/mock_confidence.json'
# path = '/home/users/yitao/Code/pytorch-mgpu-cifar10/results/check_confidence_0.01_5workers/confidence.json'

path = 'results/check_confidence_0.01_5workers/confidence.json'
# path = 'results/check_confidence_100_5workers/confidence.json'


with open(path, 'r') as f:
    data = json.load(f)

# for d in
data_np = np.asarray(data)
filtered = data_np[:, 1:]
# print(type(data_np))
# print(np.var(data_np, axis=1))

# root = 'confidence_plots/alpha_100'
root = 'confidence_plots/alpha_0.01'


names = [1, 2, 3, 4, 5]
# names = ['Edge 1', 'Edge 2', 'Edge 3', 'Edge 4', 'Edge 5']


num_row = 2
num_col = 5
sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 3})

fig, axs = plt.subplots(nrows=num_row, ncols=num_col, figsize=(50,30))
# print(type(axs))

# axs.set(xlabel='Edge', ylabel='Confidence')
# exit()
# for i, d in enumerate(data_np):
#     values = d[1:]
#     print(d[0])
#     fig = plt.figure()
#     plt.bar(names, values)
#     plt.xlabel("Edge")
#     plt.ylabel("Confidence")
#     fig.savefig(root + 'plot_' + str(i) + '.png')
    
#     if i == 9:
#         break


font_size = 40

for x, row in enumerate(axs):
    for y, col in enumerate(row):
        values = filtered[num_row * x + y]
        # print(values, names)
        col.bar(names, values)
        col.set_xlabel('Edge', fontsize=font_size)
        
        if y == 0:
            col.set_ylabel('Confidence', fontsize=font_size)
        else:
            col.set_ylabel("", fontsize=font_size)
            # col.axes.yaxis.set_ticklabels([])

        # Have to use xaxis to get rid of a warning 
        # col.xaxis.set_ticks(names)
        # col.set_xticklabels(names, fontsize=40)
        # col.set_yticklabels(fontsize=40)
        col.tick_params(axis='x', labelsize=30)
        col.tick_params(axis='y', labelsize=30)


fig.tight_layout()
fig.savefig(root + 'plot' + '.png')