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


def parse_arguments():

    parser = argparse.ArgumentParser(description="Test parameters")

    # Arguments
    parser.add_argument("--dir", default="acc.csv", type=str)
    

    args = parser.parse_args()

    return args


def collect_data(path):
    
    acc = []
    data = genfromtxt(path, delimiter=',', skip_header=1)
    length = len(data)

    for element in data:
        # add the results of all iterations
        acc.append(element)
        
    return acc


def plot(y_axis, xlabel=None, ylabel=None, title=None, output='result.png'):

    # iterate through different line styles
    lines = ["-","--",":", "-."]
    linecycler = cycle(lines)

    fontP = FontProperties()
    fontP.set_size('large')
    
    # get the key
    key = list(y_axis.keys())[0]
    # get the length of the key
    length = (len(y_axis[key]))

    x_axis = np.arange(1, length+1, 1)
    print(x_axis)

    sns.set_theme()

    # Scale elements
    # sns.set_context("paper")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2})
    
    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    
    for k, v in y_axis.items():
        ax.plot(x_axis, v, next(linecycler), label=k)
    
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height])

    # # Change the xtick freq for the iteration plot
    # plt.xticks(np.arange(0, len(x_axis)+1, 100))
    # bbox_to_anchor=(0.95, 0.15),
    
    legend_x = 0.95
    legend_y = 0
    # plt.legend(bbox_to_anchor=(0.95, 0.15), loc='lower right')
    plt.legend(bbox_to_anchor=(legend_x, legend_y), loc='lower right', ncol=2, prop=fontP)
    # Removing the top and right axes spines in the plot
    sns.despine()

    # ax.grid()
    plt.tight_layout()
    # plt.subplots_adjust(right=0.65)
    fig.savefig(output)

if __name__ == "__main__":
    args = parse_arguments()

    # lambda=1
    # edge_data = 'acc_cifar100_ten_classes_lambda_1_2021-08-06-04-39.csv'
    # cloud_data = 'distill_acc_cifar100_ten_classes_lambda_1_2021-08-06-04-39.csv'
    # edge_data = 'acc_worker1cifar100_ten_classes_lambda_1_2021-08-06-04-39.csv'
    # cloud_data = 'distill_acc_worker1_cifar100_ten_classes_lambda_1_2021-08-06-04-39.csv'

    # lambda=0.5
    # edge_data = 'acc_cifar100_ten_classes_lambda_0.5_2021-08-06-04-38.csv'
    # cloud_data = 'distill_acc_cifar100_ten_classes_lambda_0.5_2021-08-06-04-38.csv'
    edge_data = 'acc_worker1cifar100_ten_classes_lambda_0.5_2021-08-06-04-38.csv'
    cloud_data = 'distill_acc_worker1_cifar100_ten_classes_lambda_0.5_2021-08-06-04-38.csv'

    data_to_plot = {}

    data_to_plot['Edge_1'] = collect_data(edge_data)
    data_to_plot['Cloud'] = collect_data(cloud_data)

    plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Distillation with private data (ten classes)', output='lambda_0.5_w1.png')