import json
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

sns.set_theme()

def plot(data_np, root, type='confidence'):

    """
    This function takes input format as below:
    [target, norm, norm, norm, norm, norm]

    E.g., 
    [[1, 3.1185127183234256e-06, 0.04875403615767096, 0.0004979580037431997, 0.07451065717082182, 2.5734172533123304e-08], 
    [4, 5.819869786511859e-05, 1.1896488153750389, 0.013105086292398002, 0.06457583607098871, 0.18634056344432165], 
    [3, 0.06616318535487349, 0.09746917163139755, 1.938498408299161, 0.020091347707652925, 1.6364395435250942]]
    
    """
    filtered = data_np[:, 1:]
    names = [0, 1, 2, 3, 4]
    num_row = 2
    num_col = 5
    font_size = 40

    sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 3})
    fig, axs = plt.subplots(nrows=num_row, ncols=num_col, figsize=(50,30))

    for x, row in enumerate(axs):
        for y, col in enumerate(row):
            values = filtered[num_row * x + y]
            col.bar(names, values)
            # col.set_xlabel('Edge', fontsize=font_size)
            
            if y == 0:
                col.set_ylabel('Averaged norm', fontsize=font_size)
            else:
                col.set_ylabel("", fontsize=font_size)
            col.set_title('Class' + str(x * num_col + y), fontsize=font_size)
            col.tick_params(axis='x', labelsize=30)
            col.tick_params(axis='y', labelsize=30)

    fig.tight_layout()
    fig.savefig(root + 'confidence_plot_res18' + '.png')

if __name__ == '__main__':

    # path = 'results/check_confidence_0.01_5workers_last_batch/confidence.json'
    # path = 'results/check_confidence_100_5workers_last_batch/confidence.json'
    # path = 'results/check_confidence_test/confidence.json'
    # path = 'results/check_confidence_0.01_5workers/confidence.json'
    # path = 'results/check_confidence_uniform_5workers/confidence.json'
    # path = 'results/check_confidence_uniform_200ep_5workers/confidence.json'
    path = 'results/check_confidence_res18_uniform_5workers/confidence.json'

    with open(path, 'r') as f:
        data = json.load(f)

    data_np = np.asarray(data)
    print(data_np.shape)

    d_average = np.empty([10, 6])
    for i in range(10):
        mask = data_np[:, 0] == i
        d = data_np[mask]
        d_average[i] = np.average(d, axis=0)

    print(d_average)
    plot(d_average, 'confidence_plots/')