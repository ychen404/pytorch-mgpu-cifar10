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
        if 'Distillation' in k or "FD" in k and 'unlabeled' not in k:
            ax.plot(x_axis, v, next(linecycler), label=k, color='#8C1D40')
        elif 'FedAvg' in k:
            ax.plot(x_axis, v, next(linecycler), label=k, color='mediumblue')
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

    centralized_res18_20cls_50pct = 'results/res18_20cls_50_percent_save/acc_res18_2021-11-05-02-45.csv'
    centralized_res18_10cls_50pct = 'results/res18_10cls_50_percent_again/acc_res18_2021-11-18-19-22.csv'
    centralized_res18_10cls_25pct = 'results/res18_10cls_25_percent/acc_res18_2021-10-01-18-02.csv'
    centralized_res18_10cls_1_25pct = 'results/res18_10cls_12.5_percent/acc_res18_2021-11-18-19-23.csv'
    centralized_res18_10cls_100pct = 'results/res18_10cls_100_percent/acc_res18_2021-10-01-17-51.csv'
    centralized_res18_20cls_100pct = 'results/res18_20cls_100_percent_again/acc_res18_2021-11-24-17-23.csv'
    centralized_res18_10cls_100pct_lr001 = 'results/res18_20cls_100_percent_again_sgd_lr_001_again1/acc_res18_2021-11-26-17-17.csv'
    # centralized_res18_10cls_100pct_lr001 = 'results/res18_20cls_100_percent_again_sgd_lr_0001/acc_res18_2021-11-26-23-19.csv'
    
    # centralized_res18_10cls_1_25pct = 'results/res18_10cls_12.5_percent_dirichlet/acc_res18_2021-11-18-20-00.csv'

    # five workers 2 cls

    # cloud_iid_2_cls = 'results/five_workers_iid_two_cls/distill_concat_2021-09-20-22-11.csv'
    # edge_0_2_cls = 'results/five_workers_iid_two_cls/acc_worker_0_2021-09-20-22-04.csv'
    # edge_1_2_cls = 'results/five_workers_iid_two_cls/acc_worker_1_2021-09-20-22-06.csv'
    # edge_2_2_cls = 'results/five_workers_iid_two_cls/acc_worker_2_2021-09-20-22-07.csv'
    # edge_3_2_cls = 'results/five_workers_iid_two_cls/acc_worker_3_2021-09-20-22-08.csv'
    # edge_4_2_cls = 'results/five_workers_iid_two_cls/acc_worker_4_2021-09-20-22-10.csv'

   
    # there might be an easier way to populate data
    # base = 'results/two_cls_res6_five_workers_non_iid/'
    # base = 'results/non_iid_public_res8_ten_workers_2_cls_private_distill_distill_pct_0.1/'
    base = 'results/iid_ten_workers_res6_2_cls_public_distill/'

    base_public = 'results/two_cls_res6_five_workers_non_iid_public/'
    # cloud_iid_3_cls_seven = 'results/seven_workers_iid_three_cls/distill_concat_2021-09-21-17-48.csv'
    # cloud_iid_3_cls_eight = 'results/eight_workers_iid_three_cls/distill_concat_2021-09-21-17-48.csv'

    # cloud_iid_3_cls_oneshot = 'results/five_workers_iid_three_cls/distill_concat_2021-09-20-23-52.csv'
    # cloud = base + 'distill_2021-09-29-00-16.csv'
    cloud = base + 'distill_2021-11-05-04-47.csv'

    fd_with_pub = 'results/two_cls_res6_five_workers_non_iid_public/distill_2021-09-29-16-27.csv'
    cloud_pub = base_public + 'distill_2021-09-29-16-27.csv'
    cloud_10_workers = 'results/non_iid_public_res6_ten_workers_2_cls_public_distill/distill_2021-10-04-06-42.csv'
    cloud_15_workers = 'results/non_iid_public_res6_fifteen_workers_2_cls_public_distill/distill_2021-10-04-07-23.csv'


    cloud_25_pcnt = 'results/non_iid_public_res6_five_workers_2_cls_public_distill_25_2ndrun/distill_2021-10-01-22-39.csv'
    # cifar10 = base + 'cifar10_2021-09-27-17-24.csv'
    # cloud_iid_3_cls_ten = 'results/ten_workers_iid_three_cls/distill_concat_2021-09-21-17-51.csv'

    edge_0 = base + 'acc_worker_0_2021-09-28-23-59.csv'
    edge_1 = base + 'acc_worker_1_2021-09-29-00-03.csv'
    edge_2 = base + 'acc_worker_2_2021-09-29-00-06.csv'
    edge_3 = base + 'acc_worker_3_2021-09-29-00-10.csv'
    edge_4 = base + 'acc_worker_4_2021-09-29-00-13.csv'

    edge_0_local = base + 'acc_worker_0_2021-09-28-23-59_local.csv'
    edge_1_local = base + 'acc_worker_1_2021-09-29-00-03_local.csv'
    edge_2_local = base + 'acc_worker_2_2021-09-29-00-06_local.csv'
    edge_3_local = base + 'acc_worker_3_2021-09-29-00-10_local.csv'
    edge_4_local = base + 'acc_worker_4_2021-09-29-00-13_local.csv'

    # ten_pcnt = 'results/res6_2cls_10_percent/acc_res6_2021-09-29-05-11.csv'
    # twenty_pcnt = 'results/res6_2cls_20_percent/acc_res6_2021-09-29-05-13.csv'
    # fourty_pcnt = 'results/res6_2cls_40_percent/acc_res6_2021-09-29-05-15.csv'
    # sixty_pcnt = 'results/res6_2cls_60_percent/acc_res6_2021-09-29-05-18.csv'
    # eighty_pcnt = 'results/res6_2cls_80_percent/acc_res6_2021-09-29-05-20.csv'
    # hundred_pcnt = 'results/res6_2cls_100_percent/acc_res6_2021-09-29-05-23.csv'

    res6_10cls_50pcnt_alldata = 'results/res6_10cls_50_percent_dirichlet/acc_res6_2021-11-21-19-55.csv'
    res6_10cls_25pcnt_alldata = 'results/res6_10cls_25_percent_dirichlet/acc_res6_2021-11-21-19-57.csv'
    res6_10cls_32_5pcnt_alldata = 'results/res6_10cls_32.5_percent_dirichlet/acc_res6_2021-11-21-19-59.csv'
    res6_10cls_12_5_5pcnt_alldata = 'results/res6_10cls_12.5_percent_dirichlet/acc_res6_2021-11-21-21-50.csv'
    
    #cat acc.csv | awk {'print $1'} | sed 's/,//' > acc_parsed.csv
    # oneliner:    
    #cat acc.csv | awk {'print $1'} | sed 's/,//' | tail -n +2 > acc_parsed.csv
    fedavg_2cls_200round = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_2cls_200rounds_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_2cls_noniid_200round = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_non_iid_five_worker_fedavg_2cls_200rounds_resnet6_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_2cls_noniid_200round_10worker = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_non_iid_ten_worker_fedavg_2cls_200rounds_resnet6_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_2cls_iid_200round_10worker = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_ten_worker_fedavg_2cls_200rounds_resnet6_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_2cls_iid_200round_15worker_1eps = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_15_worker_fedavg_2cls_200rounds_resnet6_localep_1_eps_1_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_2cls_iid_200round_15worker = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_15_worker_fedavg_2cls_200rounds_resnet6_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_2cls_iid_200round_5worker_again = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_5_worker_fedavg_2cls_200rounds_resnet6_again_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    
    fedavg_2cls_iid_200round_10worker_1_localep = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_ten_worker_fedavg_2cls_200rounds_resnet6_localep_1_eps_1_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    
    fedavg_2cls_iid_200round_5worker = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_ten_worker_fedavg_2cls_200rounds_resnet6_localep_1_eps_1_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

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

    fedavg_10_cls_alldata = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_5_worker_fulldata_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_20_cls = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_five_worker_fedavg_20cls_200rounds_resnet6_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    
    fedavg_20_cls_alldata = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_10_worker_fulldata_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    fedavg_20_cls_75pcnt_data = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_10_worker_75pcnt_data_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    fedavg_20_cls_50pcnt_data = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_10_worker_50pcnt_data_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'
    fedavg_20_cls_25pcnt_data = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_10_worker_75pcnt_data_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'

    fedavg_10_cls_10pcnt_data = '/home/users/yitao/Code/python-socket-FL/' + \
                'results/cifar100_iid_10_worker_10cls_100_pcnt_private_data_eps_5_alpha_100.0_mode_FedAvg/cifar100/acc_parsed.csv'


    distill_5_pct = 'results/non_iid_public_res8_five_workers_2_cls_public_distill_distill_pct_0.05/distill_2021-10-28-19-18.csv'
    distill_10_pct = 'results/non_iid_public_res8_five_workers_2_cls_public_distill_distill_pct_0.1/distill_2021-10-28-19-15.csv'
    distill_20_pct = 'results/non_iid_public_res8_five_workers_2_cls_public_distill_distill_pct_0.2/distill_2021-10-28-19-15.csv'
    distill_30_pct = 'results/non_iid_public_res8_five_workers_2_cls_public_distill_distill_pct_0.3/distill_2021-10-28-19-15.csv'
    distill_40_pct = 'results/non_iid_public_res8_five_workers_2_cls_public_distill_distill_pct_0.4/distill_2021-10-28-19-22.csv'

    # private_distill_10_pct = 'results/non_iid_public_res8_five_workers_2_cls_private_distill_distill_pct_0.1/distill_2021-10-28-23-03.csv'
    # private_distill_20_pct = 'results/non_iid_public_res8_five_workers_2_cls_private_distill_distill_pct_0.2/distill_2021-10-28-22-59.csv'
    # private_distill_30_pct = 'results/non_iid_public_res8_five_workers_2_cls_private_distill_distill_pct_0.3/distill_2021-10-28-22-59.csv'
    # private_distill_40_pct = 'results/non_iid_public_res8_five_workers_2_cls_private_distill_distill_pct_0.4/distill_2021-10-28-23-07.csv'

    private_distill_10_pct = 'results/non_iid_public_res8_ten_workers_2_cls_private_distill_distill_pct_0.1/distill_2021-11-01-23-02.csv'
    private_distill_20_pct = 'results/non_iid_public_res8_ten_workers_2_cls_private_distill_distill_pct_0.2/distill_2021-11-01-22-59.csv'
    private_distill_30_pct = 'results/non_iid_public_res8_ten_workers_2_cls_private_distill_distill_pct_0.3/distill_2021-11-03-02-35.csv'
    private_distill_40_pct = 'results/non_iid_public_res8_ten_workers_2_cls_private_distill_distill_pct_0.4/distill_2021-11-03-02-32.csv'

    iid_15_workers = 'results/iid_15_workers_res6_2_cls_public_distill/distill_2021-11-08-06-14.csv'
    iid_5_workers = 'results/iid_5_workers_res6_2_cls_public_distill/distill_2021-11-08-23-41.csv'
    iid_5_workers_lambda_05_100pcnt = 'results/iid_5_workers_res6_2_cls_public_distill_lamb_0.5/distill_2021-11-09-03-57.csv'
    iid_5_workers_lambda_05_50pcnt = 'results/iid_5_workers_res6_2_cls_public_distill_lambda_0.5_50_pcnt_distill/distill_2021-11-15-15-55.csv'
    iid_5_workers_lambda_05_30pcnt = 'results/iid_5_workers_res6_2_cls_public_distill_lambda_0.5_30_pcnt_distill/distill_2021-11-15-15-54.csv'
    iid_5_workers_lambda_05_10pcnt = 'results/iid_5_workers_res6_2_cls_public_distill_lambda_0.5_10_pcnt_distill/distill_2021-11-15-15-55.csv'

    iid_5_workers_dynamic_lambda_50pcnt_alter = 'results/iid_5_workers_res6_2_cls_public_distill_dynamic_lambda_025_1_50_pcnt_distill_alter/distill_2021-11-17-23-46.csv'
    iid_5_workers_dynamic_lambda_25pcnt_alter = 'results/iid_5_workers_res6_2_cls_public_distill_dynamic_lambda_025_1_25_pcnt_distill_alter/distill_2021-11-17-23-45.csv'

    iid_5_workers_dynamic_lambda_50pcnt_alter_01 = 'results/iid_5_workers_res6_2_cls_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter/distill_2021-11-17-23-50.csv'
    iid_5_workers_dynamic_lambda_25pcnt_alter_01 = 'results/iid_5_workers_res6_2_cls_public_distill_dynamic_lambda_0_1_25_pcnt_distill_alter/distill_2021-11-18-00-06.csv'

    iid_5_workers_edge_avg_acc = 'results/iid_5_workers_res6_2_cls_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter/avg_acc.csv'

    # iid_1_workers_dynamic_lambda_50pcnt_alter_01 = 'results/iid_1_workers_res6_2_cls_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter/distill_2021-11-19-21-03.csv'
    # iid_1_workers_dynamic_lambda_50pcnt_alter_01_edge = 'results/iid_1_workers_res6_2_cls_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter/acc_worker_0_2021-11-19-21-01.csv'

    iid_1_workers_dynamic_lambda_50pcnt_alter_01 = 'results/iid_single_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_mimic_five_workers/distill_2021-11-21-19-50.csv'
    iid_1_workers_dynamic_lambda_50pcnt_alter_01_edge = 'results/iid_single_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_mimic_five_workers/acc_worker_0_2021-11-21-19-48.csv'

    iid_10_workers_dynamic_lambda_50pcnt_alter_01 = 'results/iid_10_workers_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter/distill_2021-11-19-21-21.csv'
    iid_10_workers_edge_avg_acc = 'results/iid_10_workers_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter/avg_acc.csv'
    
    iid_5_worker_all_private_data = 'results/iid_5_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_allprivatedata/distill_2021-11-22-23-22.csv'
    iid_5_worker_075_private_data = 'results/iid_5_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_random_sample_075/distill_2021-11-23-00-51.csv'
    iid_5_worker_050_private_data = 'results/iid_5_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_random_sample_050/distill_2021-11-23-00-55.csv'

    iid_10_worker_all_private_data = 'results/iid_10_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_random_sample_100/distill_2021-11-24-05-09.csv'
    iid_10_worker_075_private_data = 'results/iid_10_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_random_sample_075/distill_2021-11-24-04-52.csv'
    iid_10_worker_050_private_data = 'results/iid_10_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_random_sample_050/distill_2021-11-24-04-49.csv'
    iid_10_worker_025_private_data = 'results/iid_10_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_random_sample_025/distill_2021-11-24-04-50.csv'
    iid_10_worker_010_private_data = 'results/iid_10_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_random_sample_010/distill_2021-11-29-02-00.csv'
    # data_to_plot['Federated Distillation (res18, private, 100%data, 10cls)'] = collect_data(cloud)
    # data_to_plot['Centralized '] = collect_data(resnet18_alldata)
    # data_to_plot['Federated Distillation'] = collect_data(cloud_pub)
    # data_to_plot['100% public data to distill'] = collect_data(fd_with_pub)
    
    # data_to_plot['2.5% data distill'] = collect_data(cloud)

    # data_to_plot['10% public data to distill'] = collect_data(distill_10_pct)
    # data_to_plot['FedAvg'] = collect_data(fedavg_2cls_noniid_200round)
    # data_to_plot['20% public data to distill'] = collect_data(distill_20_pct)
    # data_to_plot['30% public data to distill'] = collect_data(distill_30_pct)
    # data_to_plot['40% public data to distill'] = collect_data(distill_40_pct)

    # data_to_plot['2.5% private distill'] = collect_data(cloud)

    ####### 10 edge worker use private data to distill
    # data_to_plot['Federated Distillation (10% shared data)'] = collect_data(private_distill_10_pct)
    # data_to_plot['Federated Distillation (20% shared data)'] = collect_data(private_distill_20_pct)
    # data_to_plot['Federated Distillation (30% shared data)'] = collect_data(private_distill_30_pct)
    # data_to_plot['Federated Distillation (40% shared data)'] = collect_data(private_distill_40_pct)
    # data_to_plot['FedAvg'] = collect_data(fedavg_2cls_noniid_200round_10worker)
    # # data_to_plot['Centralized (res18, 50%data, 20cls)'] = collect_data(centralized_res18_20cls_50pct)
    # data_to_plot['Centralized'] = collect_data(centralized_res18_20cls_50pct)

    # data_to_plot['40% private data to distill'] = collect_data(private_distill_40_pct)
    # data_to_plot['Centralized (res18, 50%data, 10cls)'] = collect_data(resnet18_halfdata)
    # data_to_plot['Federated Distillation (res18, public, 25%data, 10cls)'] = collect_data(cloud_25_pcnt)

    ########## previous non iid case 
    # data_to_plot['Federated Distillation (res18, 100%public data, 20cls)'] = collect_data(cloud_10_workers)
    # data_to_plot['FedAvg (res6, 100%private data, 20cls)'] = collect_data(fedavg_2cls_noniid_200round_10worker)
    # data_to_plot['Centralized (res18, 50%data, 20cls)'] = collect_data(centralized_res18_20cls_50pct)
    # data_to_plot['Federated Distillation'] = collect_data(cloud_10_workers)
    # data_to_plot['FedAvg'] = collect_data(fedavg_2cls_noniid_200round_10worker)
    # data_to_plot['Centralized'] = collect_data(centralized_res18_20cls_50pct)

    ########## iid case with two classes shared by all workers
    # cloud_iid_2_cls = 'results/five_workers_iid_two_cls/distill_concat_2021-09-20-22-11.csv'
    # edge_0_2_cls = 'results/five_workers_iid_two_cls/acc_worker_0_2021-09-20-22-04.csv'
    # edge_1_2_cls = 'results/five_workers_iid_two_cls/acc_worker_1_2021-09-20-22-06.csv'
    # edge_2_2_cls = 'results/five_workers_iid_two_cls/acc_worker_2_2021-09-20-22-07.csv'
    # edge_3_2_cls = 'results/five_workers_iid_two_cls/acc_worker_3_2021-09-20-22-08.csv'
    # edge_4_2_cls = 'results/five_workers_iid_two_cls/acc_worker_4_2021-09-20-22-10.csv'
    # central = 'results/res18_total_2cls_50pctdata/acc_res18_2021-11-11-03-43.csv'

    # data_to_plot['Federated Distillation (res18, iid, 100%public data, 2cls)'] = collect_data(cloud_iid_2_cls)
    # data_to_plot['FedAvg (res6, iid, 100%private data, 2cls)'] = collect_data(fedavg_2cls_200round)
    # data_to_plot['Centralized (res18, 50%data, 2cls)'] = collect_data(central)
    
    # data_to_plot['0'] = collect_data(edge_0_2_cls)
    # data_to_plot['1'] = collect_data(edge_1_2_cls)
    # data_to_plot['2'] = collect_data(edge_2_2_cls)

    ########## iid case
    # data_to_plot['Federated Distillation (res18, iid, 100%public data, 10cls)'] = collect_data(iid_5_workers)
    # data_to_plot['Federated Distillation (res18, iid, 100%public data, 20cls)'] = collect_data(cloud)
    # data_to_plot['Federated Distillation (res18, iid, 100%public data, 30cls)'] = collect_data(iid_15_workers)
    # data_to_plot['FedAvg (res6, iid, 100%private data, 10cls)'] = collect_data(fedavg_2cls_iid_200round_5worker_again)
    # data_to_plot['FedAvg (res6, iid, 100%private data, 20cls)'] = collect_data(fedavg_2cls_iid_200round_10worker)
    # data_to_plot['FedAvg (res6, iid, 100%private data, 30cls)'] = collect_data(fedavg_2cls_iid_200round_15worker)
    
    ########## iid case lambda 0.5 and dynamic case
    # iid_5_workers_res6_2_cls_public_distill_lambda_0.5_50_pcnt_distill
    # data_to_plot['Federated Distillation (res18, iid, 100% public data, lambda 0.5, 10cls)'] = collect_data(iid_5_workers_lambda_05_100pcnt)
    # data_to_plot['Federated Distillation (res18, iid, 50% public data, lambda 0.5, 10cls)'] = collect_data(iid_5_workers_lambda_05_50pcnt)
    # data_to_plot['Federated Distillation (res18, iid, 30% public data, lambda 0.5, 10cls)'] = collect_data(iid_5_workers_lambda_05_30pcnt)
    # data_to_plot['Federated Distillation (res18, iid, 10% public data, lambda 0.5, 10cls)'] = collect_data(iid_5_workers_lambda_05_10pcnt)
    # data_to_plot['Federated Distillation (res18, iid, 100% public data, unlabeled, 10cls)'] = collect_data(iid_5_workers)
    
    #### dynamic cases ####
    # data_to_plot['Federated Distillation (res18, iid, dynamic lambda(0.75, 1) 10%public data, 10cls)'] = collect_data(iid_5_workers_dynamic_lambda_10pcnt)
    # data_to_plot['Federated Distillation (res18, iid, lambda(0.25 & 1) 50%public data, alt, 10cls)'] = collect_data(iid_5_workers_dynamic_lambda_50pcnt_alter)
    # data_to_plot['Federated Distillation (res18, iid, lambda(0.25 & 1) 25%public data, alt, 10cls)'] = collect_data(iid_5_workers_dynamic_lambda_25pcnt_alter)
    
    # data_to_plot['Federated Distillation (res18, 5 worker, iid, lambda(0 & 1) 50%public data, alt, 10cls)'] = collect_data(iid_5_workers_dynamic_lambda_50pcnt_alter_01)
    # data_to_plot['Federated Distillation (res18, iid, lambda(0 & 1) 25%public data, alt, 10cls)'] = collect_data(iid_5_workers_dynamic_lambda_25pcnt_alter_01)
    # data_to_plot['Federated Distillation (res18, 10 worker, iid, lambda(0 & 1) 50%public data, alt, 10cls)'] = collect_data(iid_10_workers_dynamic_lambda_50pcnt_alter_01)

    # data_to_plot['Average Accuracy of 10 workers'] = collect_data(iid_10_workers_edge_avg_acc)
    # data_to_plot['Average Accuracy of 5 workers'] = collect_data(iid_5_workers_edge_avg_acc)
    # data_to_plot['Federated Distillation (res18, 10 worker, iid, lambda(0 & 1) 50%public data, alt, 10cls)'] = collect_data(iid_1_workers_dynamic_lambda_50pcnt_alter_01)
    # data_to_plot['Edge (res6, 20%private data, 10cls)'] = collect_data(iid_1_workers_dynamic_lambda_50pcnt_alter_01_edge)
    # data_to_plot['Centralized (res18, 50%public data, 10cls)'] = collect_data(centralized_res18_10cls_25pct)
    # data_to_plot['Centralized (res18, 25%public data, 10cls)'] = collect_data(centralized_res18_10cls_1_25pct)

    #### overlapped edge data 5 workers
    # data_to_plot["FD (Each edge has all private data)"] = collect_data(iid_5_worker_all_private_data)
    # # data_to_plot["FD (Each edge has 75% private data)"] = collect_data(iid_5_worker_075_private_data)
    # # data_to_plot["FD (Each edge has 50% private data)"] = collect_data(iid_5_worker_050_private_data)
    # data_to_plot['FedAvg (res6, iid, Each edge has all private data, 10cls)'] = collect_data(fedavg_10_cls_alldata)
    # data_to_plot['Centralized '] = collect_data(resnet18_alldata)

    #### overlapped edge data 10 workers
    # data_to_plot["Federated Distillation (100% private data)"] = collect_data(iid_10_worker_all_private_data)
    # data_to_plot["Federated Distillation (75% private data)"] = collect_data(iid_10_worker_075_private_data)
    # data_to_plot["Federated Distillation (50% private data)"] = collect_data(iid_10_worker_050_private_data)
    # data_to_plot["Federated Distillation (25% private data)"] = collect_data(iid_10_worker_025_private_data)
    # data_to_plot["FedAvg (100% private data)"] = collect_data(fedavg_20_cls_alldata)
    # data_to_plot["FedAvg (75% private data)"] = collect_data(fedavg_20_cls_75pcnt_data)
    # data_to_plot["FedAvg (50% private data)"] = collect_data(fedavg_20_cls_50pcnt_data)
    # data_to_plot["FedAvg (25% private data)"] = collect_data(fedavg_20_cls_25pcnt_data)

    ## 11/30
    # data_to_plot["Federated Distillation"] = collect_data(iid_10_worker_010_private_data)
    # data_to_plot["FedAvg"] = collect_data(fedavg_10_cls_10pcnt_data)
    # data_to_plot['Centralized '] = collect_data(centralized_res18_10cls_100pct_lr001)
    
    ############# plot edge
    # data_to_plot["100% private data"] = collect_data(res6_10cls_50pcnt_alldata)
    # data_to_plot["75% private data"] = collect_data(res6_10cls_32_5pcnt_alldata)
    # data_to_plot["50% private data"] = collect_data(res6_10cls_25pcnt_alldata)
    # data_to_plot["25% private data"] = collect_data(res6_10cls_12_5_5pcnt_alldata)

    # plot normal
    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', '', output= base  + 'iid_partial_distill.png')
    
    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', '', output= 'results/non_iid_public_res8_ten_workers_2_cls_private_distill_distill_pct_0.1/'  + 'partial_distill.png')
    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', '', output= 'results/non_iid_public_res6_ten_workers_2_cls_public_distill/'  + 'partial_distill.png')
    plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', '', output= 'results/iid_5_worker_res6_public_distill_dynamic_lambda_0_1_50_pcnt_distill_alter_data_each_edge_allprivatedata/'  + 'result.png')