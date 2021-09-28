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

    # five workers 2 cls

    # cloud_iid_2_cls = 'results/five_workers_iid_two_cls/distill_concat_2021-09-20-22-11.csv'
    # edge_0_2_cls = 'results/five_workers_iid_two_cls/acc_worker_0_2021-09-20-22-04.csv'
    # edge_1_2_cls = 'results/five_workers_iid_two_cls/acc_worker_1_2021-09-20-22-06.csv'
    # edge_2_2_cls = 'results/five_workers_iid_two_cls/acc_worker_2_2021-09-20-22-07.csv'
    # edge_3_2_cls = 'results/five_workers_iid_two_cls/acc_worker_3_2021-09-20-22-08.csv'
    # edge_4_2_cls = 'results/five_workers_iid_two_cls/acc_worker_4_2021-09-20-22-10.csv'

    # six workers

    # cloud_iid_3_cls = 'results/five_workers_iid_three_cls_res6/distill_concat_2021-09-22-20-26.csv'
    # edge_0_3_cls = 'results/five_workers_iid_three_cls_res6/acc_worker_0_2021-09-22-20-19.csv'
    # edge_1_3_cls = 'results/five_workers_iid_three_cls_res6/acc_worker_1_2021-09-22-20-21.csv'
    # edge_2_3_cls = 'results/five_workers_iid_three_cls_res6/acc_worker_2_2021-09-22-20-22.csv'
    # edge_3_3_cls = 'results/five_workers_iid_three_cls_res6/acc_worker_3_2021-09-22-20-23.csv'
    # edge_4_3_cls = 'results/five_workers_iid_three_cls_res6/acc_worker_4_2021-09-22-20-25.csv'
    # edge_5_3_cls = 'results/six_workers_iid_three_cls/acc_worker_5_2021-09-21-00-07.csv'

    
    # there might be an easier way to populate data
    base = 'results/seven_workers_iid_four_cls_res6/'
    # cloud_iid_3_cls_seven = 'results/seven_workers_iid_three_cls/distill_concat_2021-09-21-17-48.csv'
    # cloud_iid_3_cls_eight = 'results/eight_workers_iid_three_cls/distill_concat_2021-09-21-17-48.csv'

    # cloud_iid_3_cls_oneshot = 'results/five_workers_iid_three_cls/distill_concat_2021-09-20-23-52.csv'
    cloud_iid = base + 'distill_2021-09-28-06-00.csv'

    # cifar10 = base + 'cifar10_2021-09-27-17-24.csv'
    # cloud_iid_3_cls_ten = 'results/ten_workers_iid_three_cls/distill_concat_2021-09-21-17-51.csv'

    edge_0 = base + 'acc_worker_0_2021-09-28-05-51.csv'
    edge_1 = base + 'acc_worker_1_2021-09-28-05-52.csv'
    edge_2 = base + 'acc_worker_2_2021-09-28-05-53.csv'
    edge_3 = base + 'acc_worker_3_2021-09-28-05-55.csv'
    edge_4 = base + 'acc_worker_4_2021-09-24-17-59.csv'

    edge_5 = base + 'acc_worker_5_2021-09-28-05-57.csv'
    edge_6 = base + 'acc_worker_6_2021-09-28-05-59.csv'
    
    # finetune = base + 'acc_worker_9_2021-09-27-18-05.csv'
    # FD_bound = 'results/res18_20cls/acc_res18_2021-09-27-19-16.csv'
    # FedAvg_bound = 'results/res6_20cls/acc_res6_2021-09-27-19-16.csv'

    
    # edge_5_3_cls = base + 'acc_worker_5.csv'
    
    # edge_6_3_cls = base + 'acc_worker_6_2021-09-21-17-45.csv'
    # edge_7_3_cls = base + 'acc_worker_7_2021-09-21-17-46.csv'
    
    #cat acc.csv | awk {'print $1'} | sed 's/,//' > acc_parsed.csv
    fedavg_2cls_200round = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_2cls_200rounds_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_3_cls_200round = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_six_worker_fedavg_3cls_200rounds_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    
    fedavg_6_cls_200round = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_6cls_200rounds_resnet6_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_3_cls_6_workers_200round = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_six_worker_fedavg_3cls_200rounds_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_3_cls_7_workers_200round = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_seven_worker_fedavg_3cls_200rounds_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_3_cls_8_workers_200round = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_eight_worker_fedavg_3cls_200rounds_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    
    fedavg_5_cls = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_eps_10_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    
    fedavg_10_cls = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_10cls_eps_10_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_20_cls = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_20cls_200rounds_resnet6_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    # data_to_plot['Federated Distillation 9 workers'] = collect_data(cloud_iid_3_cls_nine)
    # data_to_plot['Federated Distillation 10 workers'] = collect_data(cloud_iid_3_cls_ten)
    # data_to_plot['Federated Distillation 7 workers'] = collect_data(cloud_iid_3_cls_seven)
    # data_to_plot['Federated Distillation 8 workers'] = collect_data(cloud_iid_3_cls_eight)

    data_to_plot['Federated Distillation'] = collect_data(cloud_iid)
    # data_to_plot['Federated Distillation One-Shot'] = collect_data(cloud_iid_3_cls_oneshot)
    data_to_plot['Edge 0'] = collect_data(edge_0)
    data_to_plot['Edge 1'] = collect_data(edge_1)
    data_to_plot['Edge 2'] = collect_data(edge_2)
    data_to_plot['Edge 3'] = collect_data(edge_3)
    data_to_plot['Edge 4'] = collect_data(edge_4)
    # data_to_plot['Edge 5'] = collect_data(edge_5_3_cls)
    # data_to_plot['Edge 6'] = collect_data(edge_6_3_cls)
    data_to_plot['FedAvg'] = collect_data(fedavg_20_cls)
    # data_to_plot['Add Cifar10'] = collect_data(cifar10)
    # data_to_plot['Finetune'] = collect_data(finetune)
    data_to_plot['FD Upper Bound'] = collect_data(FD_bound)
    data_to_plot['FedAvg Upper Bound'] = collect_data(FedAvg_bound)


    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Both Edge and Cloud Classifies 10 Classes', output='results/five_workers_iid/iid.png')
    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Each Edge classifies 2 Classes', output='results/public_percent_0.5_2_cls_adam_lambda_1/edge_full.png')
    plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', '', output=base + 'iid_w_bound.png')
