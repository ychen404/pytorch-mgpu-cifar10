import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import seaborn as sns
from itertools import cycle
from matplotlib.font_manager import FontProperties
import sys
from matplotlib.pyplot import text

# ws = sys.argv[1]
# distill_result = sys.argv[2]
# finetune_result = sys.argv[3]

# print("Argument List:", str(sys.argv))

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
    fontP.set_size('medium')
    
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
        if 'Distillation' in k:
            ax.plot(x_axis, v, next(linecycler), label=k, color='maroon')
        elif 'FedAvg' in k:
            ax.plot(x_axis, v, next(linecycler), label=k, color='mediumblue')
        else:
            ax.plot(x_axis, v, next(linecycler), label=k, color='green')
    
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height])

    # # Change the xtick freq for the iteration plot
    # plt.xticks(np.arange(0, len(x_axis)+1, 100))
    # bbox_to_anchor=(0.95, 0.15),
    
    legend_x = 1
    legend_y = 0
    # plt.legend(bbox_to_anchor=(0.95, 0.15), loc='lower right')
    plt.legend(bbox_to_anchor=(legend_x, legend_y), loc='lower right', ncol=1, prop=fontP)

    # draw a line to show finetune
    plt.axvline(x=200, color='black', ls='--', label='finetune')
    text(190, 45, "Finetune starts", rotation=90, verticalalignment='center')

    # Removing the top and right axes spines in the plot
    sns.despine()

    # ax.grid()
    plt.tight_layout()
    # plt.subplots_adjust(right=0.65)
    fig.savefig(output)

path_cloud = 'distill_2021-11-12-16-35.csv'
path = 'cloud_finetune.csv'
ws = 'results/iid_5_workers_res6_2_cls_public_distill_finetune_10eps_100_publicdata/'

# acc_cloud = collect_data(ws + distill_result)
# acc = collect_data(ws + finetune_result)

acc_cloud = collect_data(ws + path_cloud)
acc = collect_data(ws + path)
all_acc = acc_cloud + acc

# distill_public_data_30_pcnt = collect_data('results/iid_5_workers_res6_2_cls_public_distill_finetune_50eps_0.3_publicdata/distill_2021-11-12-17-06.csv')
# finetune_public_data_30_pcnt = collect_data('results/iid_5_workers_res6_2_cls_public_distill_finetune_50eps_0.3_publicdata/cloud_finetune.csv')
# all_30_pcnt = distill_public_data_30_pcnt + finetune_public_data_30_pcnt

# distill_public_data_100_pcnt = collect_data('results/iid_5_workers_res6_2_cls_public_distill_finetune_50eps_100_publicdata/distill_2021-11-12-16-34.csv')
# finetune_public_data_100_pcnt = collect_data('results/iid_5_workers_res6_2_cls_public_distill_finetune_50eps_100_publicdata/cloud_finetune.csv')
# all_100_pcnt = distill_public_data_100_pcnt + finetune_public_data_100_pcnt

distill_public_data_100_pcnt = collect_data('results/iid_5_workers_res6_2_cls_public_distill_finetune_alllayers_50eps_all_publicdata/distill_2021-11-12-22-57.csv')
finetune_public_data_100_pcnt = collect_data('results/iid_5_workers_res6_2_cls_public_distill_finetune_alllayers_50eps_all_publicdata/cloud_finetune.csv')
all_100_pcnt = distill_public_data_100_pcnt + finetune_public_data_100_pcnt

plot_data = {}
# plot_data["Federated Distillation and Finetune"] = all_acc
# plot_data["30 percent public data"] = all_30_pcnt
plot_data["100 percent public data all layers"] = all_100_pcnt
plot(plot_data, xlabel='Iteration', ylabel='Accuracy', output='results/iid_5_workers_res6_2_cls_public_distill_finetune_alllayers_50eps_all_publicdata/result.png')
