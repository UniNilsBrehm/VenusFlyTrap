import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed


def alternate_plus_minus(f_size, start_with='minus'):
    f = [0] * f_size
    if start_with == 'minus':
        f_even = -1
        f_odd = 1
    else:
        f_even = 1
        f_odd = -1

    for kk, vv in enumerate(f):
        if kk % 2 == 0:
            f[kk] = int(f_even)
        else:
            f[kk] = int(f_odd)
    return f


def plot_scatter_plot(ax, data):
    x = np.arange(1, len(data)+1, 1)
    for k, m in enumerate(data):
        a, b = 0.02, 0.15
        c = (b - a) * np.random.random_sample(len(m)) + a
        xx = [x[k]] + (c * alternate_plus_minus(len(m)))

        ax.scatter(xx, m, s=5)
    # ax.set_xticks(x, labels, rotation=30)


def plot_box_plot(ax, s, data):
    bplot = ax.boxplot(data, showfliers=False, whis=True, notch=False, bootstrap=10000, patch_artist=False)
    # Change Median Color and Line Width ((['whiskers', 'caps', 'boxes', 'medians', 'fliers', 'means']))
    for pc in bplot['medians']:
        # pc.set_color(s.BoxPlotColor)
        pc.set_color('tab:red')
        pc.set_linewidth(s.BoxPlotLw)
    for pc in bplot['boxes']:
        pc.set_color(s.BoxPlotColor)
        pc.set_linewidth(s.BoxPlotLw)
        # pc.set_facecolor('gray')
        # pc.set_edgecolor('black')
        # pc.set_hatch('x')
    for pc in bplot['whiskers']:
        pc.set_color(s.BoxPlotColor)
        pc.set_linewidth(s.BoxPlotLw)


data_files = [
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231026/data_01/20231026_data_01_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231026/data_02/20231026_data_02_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231026/data_03/20231026_data_03_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231027/data_04/20231027_data_04_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231027/data_06/20231027_data_06_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231027/data_07/20231027_data_07_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231030/data_08/20231030_data_08_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231030/data_09/20231030_data_09_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231030/data_10/20231030_data_10_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231030/data_11/20231030_data_11_results.csv',
    'C:/UniFreiburg/VenusFlyTrap/recordings/20231030/data_12/20231030_data_12_results.csv',
]

results = pd.DataFrame()
for f in data_files:
    res = pd.read_csv(f)
    results = pd.concat([results, res])

results.reset_index(inplace=True, drop=True)
results.to_csv('C:/UniFreiburg/VenusFlyTrap/recordings/all_results.csv', index=False)

plotting_data = dict()
n = []
for p in ['deltaForce', 'Latency', 'TimeConstant']:
    box_plot_data = []
    # for k in range(0, results['ID'].max()+1):
    # Ignore number 7 (the last one)
    for k in range(0, results['ID'].max()):
        dummy = results[results['ID'] == k]
        box_plot_data.append(dummy[p].to_numpy())
        n.append(dummy.shape[0])
    plotting_data[p] = box_plot_data

fig, axs = plt.subplots(1, 3)
fig.set_size_inches(15, 5, forward=True)
fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.2, hspace=0.2)

plot_scatter_plot(axs[0], plotting_data['deltaForce'])
axs[0].set_title(f'n = {n[0:6]}')
axs[0].boxplot(plotting_data['deltaForce'], showfliers=False, whis=True, notch=False, bootstrap=10000, patch_artist=False)
axs[0].set_xlabel('Action Potential Pair')
axs[0].set_ylabel('Force [N]')

plot_scatter_plot(axs[1], plotting_data['Latency'])
axs[1].set_title(f'n = {n[0:6]}')
axs[1].boxplot(plotting_data['Latency'], showfliers=False, whis=True, notch=False, bootstrap=10000, patch_artist=False)
axs[1].set_xlabel('Action Potential Pair')
axs[1].set_ylabel('Latency [s]')

plot_scatter_plot(axs[2], plotting_data['TimeConstant'])
axs[2].set_title(f'n = {n[0:6]}')
axs[2].boxplot(plotting_data['TimeConstant'], showfliers=False, whis=True, notch=False, bootstrap=10000, patch_artist=False)
axs[2].set_xlabel('Action Potential Pair')
axs[2].set_ylabel('Time Constant [s]')

plt.show()
plt.tight_layout()

# results[results['ID'] == 0]
embed()
exit()