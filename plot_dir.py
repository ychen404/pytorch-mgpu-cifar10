import matplotlib
import argparse
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import seaborn as sns
import os
import csv
from itertools import cycle
import pdb
from matplotlib.font_manager import FontProperties

def collect_data(path):
    
    acc = []
    data = genfromtxt(path, delimiter=',')
    length = len(data)

    for element in data:
        # add the results of all iterations
        acc.append(element)
        
    return acc

def plot(y_axis, xlabel=None, ylabel=None, title=None, set_color='r', output='result.png'):

    # iterate through different line styles
    lines = ["-","--",":", "-."]
    linecycler = cycle(lines)

    fontP = FontProperties()
    # fontP.set_size('small')
    
    # get the key
    key = list(y_axis.keys())[0]
    # get the length of the key
    length = (len(y_axis[key]))
    x_axis = np.arange(1, length+1, 1)
    print(x_axis)

    sns.set_theme()

    # Scale elements
    # sns.set_context("paper")
    sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 3})
    
    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    
    same_color = False

    for k, v in y_axis.items():
        if 'Distillation' in k or "FD" in k and 'unlabeled' not in k:
            ax.plot(x_axis, v, next(linecycler), label=k, color='#8C1D40')
        elif 'FedAvg' in k:
            if same_color:
                ax.plot(x_axis, v, next(linecycler), label=k, color='mediumblue')
            else:
                ax.plot(x_axis, v, next(linecycler), label=k)
        else:
            ax.plot(x_axis, v, ls='-', label=k, color='darkgreen')
    
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height])

    # # Change the xtick freq for the iteration plot
    # plt.xticks(np.arange(0, len(x_axis)+1, 100))
    # bbox_to_anchor=(0.95, 0.15),
    
    legend_x = 1
    legend_y = 0
    # plt.legend(bbox_to_anchor=(0.95, 0.15), loc='lower right')
    plt.legend(bbox_to_anchor=(legend_x, legend_y), loc='lower right', ncol=1, prop=fontP)
    # Removing the top and right axes spines in the plot
    sns.despine()

    # ax.grid()
    plt.tight_layout()
    # plt.subplots_adjust(right=0.65)
    fig.savefig(output)

if __name__ == "__main__":
    
    
    data_to_plot = {}
    fedavg_3worker_alpha_100 = 'results/emb_diri_dlc_10cls_fedavg_alpha_100/acc_fedavg.csv'
    fedavg_3worker_alpha_1 = 'results/emb_diri_dlc_10cls_fedavg_alpha_1/acc_fedavg.csv'
    fedavg_3worker_alpha_001 = 'results/emb_diri_dlc_10cls_fedavg_alpha_0.01/acc_fedavg.csv'


    ########## previous non iid case 
    data_to_plot['FedAvg (res6, alpha=100, 3 edge workers)'] = collect_data(fedavg_3worker_alpha_100)
    data_to_plot['FedAvg (res6, alpha=1, 3 edge workers)'] = collect_data(fedavg_3worker_alpha_1)
    data_to_plot['FedAvg (res6, alpha=0.01, 3 edge workers)'] = collect_data(fedavg_3worker_alpha_001)


    plot(data_to_plot, 'Round', 'Top-1 test accuracy', '', output= 'results/emb_diri_dlc_10cls_fedavg_alpha_100/'  + 'result.png')