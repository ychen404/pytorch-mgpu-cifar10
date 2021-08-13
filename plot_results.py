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
    fontP.set_size('small')
    
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
    
    #baseline results
    # edge_data_base_20cls = 'results/useful/baselines_edge_20_classes/acc_baseline_first_20_classes_res8_lambda_1_worker_0_2021-08-09-21-59.csv'
    # cloud_data_base_20cls = 'results/useful/baselines_edge_20_classes/acc_baseline_first_20_classes_res50_lambda_1_worker_0_2021-08-09-22-01.csv'
    # # edge_data_full = 'single_model_training_results/acc_cifar100_fulldata_res8_2021-07-29-22-54.csv'
    # # cloud_data_full = 'single_model_training_results/acc_cifar100_fulldata_res50_2021-07-30-22-39.csv'
    # dependent_cloud = 'results/alternate_data_res18/distill_alternate_2021-08-11-00-28.csv'
    edge_data_base_30cls = 'results/res8_30cls_baseline/acc_worker_0_2021-08-11-18-04.csv'
    dependent_cloud = 'results/alternate_data_3_workers_res18/distill_alternate_2021-08-11-17-03.csv'
    dependent_cloud_res20 = 'results/alternate_data_3_workers_res20/distill_alternate_2021-08-11-18-59.csv'
    dependent_cloud_adam = 'results/alternate_data_3_workers_res18_adam_1/distill_alternate_2021-08-13-00-06.csv'
    cloud_data_base_30cls = 'results/res18_30cls_baseline/acc_worker_0_2021-08-11-18-13.csv'


    edge_0_data = 'results/alternate_data_3_workers_res18/acc_worker_0_2021-08-11-16-52.csv'
    edge_1_data = 'results/alternate_data_3_workers_res18/acc_worker_1_2021-08-11-16-56.csv'
    edge_2_data = 'results/alternate_data_3_workers_res18/acc_worker_2_2021-08-11-16-59.csv'

    data_to_plot = {}

    # data_to_plot['Edge baseline'] = collect_data(edge_data_base_30cls)
    # data_to_plot['Cloud baseline'] = collect_data(cloud_data_base_30cls)
    # data_to_plot['Dependent cloud'] = collect_data(dependent_cloud)
    # data_to_plot['Dependent cloud Res20'] = collect_data(dependent_cloud_res20)
    # data_to_plot['Dependent cloud Adam'] = collect_data(dependent_cloud_adam)

    data_to_plot['edge_0'] = collect_data(edge_0_data)
    data_to_plot['edge_1'] = collect_data(edge_1_data)
    data_to_plot['edge_2'] = collect_data(edge_2_data)

    plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Accuracy of 30 classes', output='acc_30cls_w_workers.png')