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
    
    legend_x = 1
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
    
    data_to_plot = {}
    cloud = 'results/public_percent_0.5_2_cls_adam_lambda_1/distill_concat_2021-09-14-17-46.csv'
    edge_0 = 'results/public_percent_0.5_2_cls_adam_lambda_1/acc_worker_0_2021-09-14-17-42.csv'
    edge_1 = 'results/public_percent_0.5_2_cls_adam_lambda_1/acc_worker_1_2021-09-14-17-44.csv'
    edge_0_full = 'results/full_data_2_cls_adam_lambda_1/acc_worker_0_2021-09-14-19-12.csv'
    edge_1_full = 'results/full_data_2_cls_adam_lambda_1/acc_worker_1_2021-09-14-19-14.csv'
    cloud_private = 'results/public_percent_0.5_2_cls_adam_lambda_1_private_distill/distill_concat_2021-09-14-18-37.csv'
    cloud_full = 'results/full_data_2_cls_adam_lambda_1/distill_concat_2021-09-14-19-17.csv'

    edge_0_iid = 'results/public_percent_0.5_4_cls_adam_lambda_1_iid_public_distill_again/acc_worker_0_2021-09-15-15-48.csv'
    edge_1_iid = 'results/public_percent_0.5_4_cls_adam_lambda_1_iid_public_distill_again/acc_worker_1_2021-09-15-15-50.csv'
    cloud_iid = 'results/public_percent_0.5_4_cls_adam_lambda_1_iid_public_distill_again/distill_concat_2021-09-15-15-51.csv'


    cloud_iid_5_cls = 'results/five_workers_iid/distill_concat_2021-09-17-06-47.csv'
    edge_0_5_cls = 'results/five_workers_iid/acc_worker_0_2021-09-17-06-37.csv'
    edge_1_5_cls = 'results/five_workers_iid/acc_worker_1_2021-09-17-06-39.csv'
    edge_2_5_cls = 'results/five_workers_iid/acc_worker_2_2021-09-17-06-40.csv'
    edge_3_5_cls = 'results/five_workers_iid/acc_worker_3_2021-09-17-06-43.csv'
    edge_4_5_cls = 'results/five_workers_iid/acc_worker_4_2021-09-17-06-45.csv'

    fedavg_5_cls = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_eps_10_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    
    fedavg_10_cls = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_10cls_eps_10_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    cloud_iid_10_cls = 'results/five_workers_iid_ten_cls/distill_concat_2021-09-17-06-54.csv'
    edge_0_10_cls = 'results/five_workers_iid_ten_cls/acc_worker_0_2021-09-17-06-46.csv'
    edge_1_10_cls = 'results/five_workers_iid_ten_cls/acc_worker_1_2021-09-17-06-47.csv'
    edge_2_10_cls = 'results/five_workers_iid_ten_cls/acc_worker_2_2021-09-17-06-49.csv'
    edge_3_10_cls = 'results/five_workers_iid_ten_cls/acc_worker_3_2021-09-17-06-51.csv'
    edge_4_10_cls = 'results/five_workers_iid_ten_cls/acc_worker_4_2021-09-17-06-53.csv'

    # data_to_plot['Cloud Public'] = collect_data(cloud)
    # data_to_plot['Cloud Private'] = collect_data(cloud_private)
    # data_to_plot['Cloud 100% data'] = collect_data(cloud_full)

    # data_to_plot['Cloud'] = collect_data(cloud_iid)
    # data_to_plot['Edge 0'] = collect_data(edge_0_iid)
    # data_to_plot['Edge 1'] = collect_data(edge_1_iid)

    # data_to_plot['Edge 0 50% data'] = collect_data(edge_0)
    # data_to_plot['Edge 1 50% data'] = collect_data(edge_1)
    # data_to_plot['Edge 0 100% data'] = collect_data(edge_0_full)
    # data_to_plot['Edge 1 100% data'] = collect_data(edge_1_full)

    # data_to_plot['Distilled Cloud'] = collect_data(cloud_iid_10_cls)
    # data_to_plot['Edge 0'] = collect_data(edge_0_10_cls)
    # data_to_plot['Edge 1'] = collect_data(edge_1_10_cls)
    # data_to_plot['Edge 2'] = collect_data(edge_2_10_cls)
    # data_to_plot['Edge 3'] = collect_data(edge_3_10_cls)
    # data_to_plot['Edge 4'] = collect_data(edge_4_10_cls)

    # data_to_plot['FedAvg'] = collect_data(fedavg_5_cls)
    data_to_plot['FedAvg 10cls'] = collect_data(fedavg_10_cls)

    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Both Edge and Cloud Classifies 10 Classes', output='results/five_workers_iid/iid.png')
    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Each Edge classifies 2 Classes', output='results/public_percent_0.5_2_cls_adam_lambda_1/edge_full.png')
    plot(data_to_plot, 'Round', 'Top-1 test accuracy', 'FedAvg 5 workers 10 Classes', output='results/five_workers_iid/fedavg_10cls.png')
