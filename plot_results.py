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
    cloud_data_res18_labmda_05 = 'results/alternate_data_3_workers_res18_adam_lambda_0.5/distill_alternate_2021-08-16-16-53.csv'
    cloud_data_res18_lambda_02 = 'results/alternate_data_3_workers_res18_adam_lambda_0.2/distill_alternate_2021-08-16-19-00.csv'
    cloud_data_res18_lambda_0 = 'results/alternate_data_3_workers_res18_adam_lambda_0/distill_alternate_2021-08-16-22-42.csv'
    edge_0_data = 'results/alternate_data_3_workers_res18/acc_worker_0_2021-08-11-16-52.csv'
    edge_1_data = 'results/alternate_data_3_workers_res18/acc_worker_1_2021-08-11-16-56.csv'
    edge_2_data = 'results/alternate_data_3_workers_res18/acc_worker_2_2021-08-11-16-59.csv'
    cloud_data_res18_lambda_0_baseline = 'results/baseline_only_true_label_concat_data_3_workers_res18_adam_lambda_0/distill_concat_2021-08-27-05-12.csv'

    # 8/24
    cloud_data_lambda_0_5_average = 'results/concat_data_3_workers_average_softmax_lambda_0.5/distill_concat_2021-08-24-00-07.csv'
    cloud_data_lambda_0_5_weighted = 'results/concat_data_3_workers_average_softmax_lambda_weighted_0.5/distill_concat_2021-08-24-06-31.csv'

    iid_cloud_baseline = 'results/iid_cloud_baseline/acc_worker_0_2021-08-24-16-42.csv'
    # iid_edge_baseline = 'results/iid_edge_baseline/acc_worker_0_2021-08-24-16-41.csv'
    iid_edge_baseline = 'results/iid_cloud_baseline_use_first_again/acc_worker_0_2021-08-25-16-23.csv'

    # 8/25
    cloud_public_distill_lambda_0_5 = 'results/public_data_distill_3_workers_lambda_0.5/distill_concat_2021-08-25-04-35.csv'

    # 8/26
    public_data_distill_lambda_1 = 'results/public_data_distill_3_workers_lambda_1/distill_concat_2021-08-25-21-08.csv'
    concat_data_lambda_1 = 'results/concat_data_3_workers_res18_adam_lambda_1/distill_concat_2021-08-26-06-46.csv'
    data_selection_distill_3_workers_lambda_0_5 = 'results/data_selection_distill_3_workers_lambda_0.5/distill_concat_2021-08-26-06-35.csv'
    public_data_lambda_0 = 'results/public_data_3_workers_res18_adam_lambda_0/distill_concat_2021-08-26-19-18.csv'
    data_to_plot = {}

    # 8/27
    selection_private_data_lambda_1 = 'results/selection_private_data_3_workers_res18_adam_lambda_1/distill_concat_2021-08-27-05-13.csv'

    # data_to_plot['Cloud baseline'] = collect_data(cloud_data_base_30cls)
    # data_to_plot['Dependent cloud'] = collect_data(dependent_cloud)
    # data_to_plot['Dependent cloud Res20'] = collect_data(dependent_cloud_res20)
    # data_to_plot['No true label (lambda=1)'] = collect_data(dependent_cloud_adam)

    # data_to_plot['edge_0'] = collect_data(edge_0_data)
    # data_to_plot['edge_1'] = collect_data(edge_1_data)
    # data_to_plot['edge_2'] = collect_data(edge_2_data)

    # data_to_plot['lambda=0.5'] = collect_data(cloud_data_res18_labmda_05)
    # data_to_plot['lambda=0.2'] = collect_data(cloud_data_res18_lambda_02)

    data_to_plot['Only True label (lambda=0 baseline)'] = collect_data(cloud_data_res18_lambda_0_baseline)


    data_to_plot['Private data lambda=0.5'] = collect_data(cloud_data_lambda_0_5_weighted)
    # data_to_plot['lambda=0.5 average'] = collect_data(cloud_data_lambda_0_5_average)
    data_to_plot['Public data lambda=0.5'] = collect_data(cloud_public_distill_lambda_0_5)
        
    data_to_plot['Public data lambda=1'] = collect_data(public_data_distill_lambda_1)
    data_to_plot['Private data lambda=1'] = collect_data(concat_data_lambda_1)
    # data_to_plot['Public data lambda=0'] = collect_data(public_data_lambda_0)

    data_to_plot['Edge baseline'] = collect_data(edge_data_base_30cls)
    data_to_plot['Private data lambda=1 (selection)'] = collect_data(selection_private_data_lambda_1)

    # data_to_plot['Private data lambda=0.5 (selection)'] = collect_data(data_selection_distill_3_workers_lambda_0_5)

    # data_to_plot['lambda=0.5 alternating batches'] = collect_data(cloud_data_res18_labmda_05)

    # data_to_plot['IID cloud baseline'] = collect_data(iid_cloud_baseline)
    # data_to_plot['IID edge baseline'] = collect_data(iid_edge_baseline)


    plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Accuracy of 30 classes', output='acc_30cls_add_selection_lamb_1.png')
    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Accuracy of 30 classes', output='acc_30cls_distill_weighted_0826.png')

    # plot(data_to_plot, 'Iteration', 'Top-1 test accuracy', 'Accuracy of 30 classes (IID)', output='acc_30cls_iid_new.png')