import os
import os.path as osp
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.utils import load_time_dir


def main():
    dir = '/home/erschultz/timing_analysis'
    m_list = [256, 512, 1024]
    times_dict = defaultdict(list) # m : list of total times
    gnn_times_dict = defaultdict(list) # m : list of total times
    samples = range(1, 16)
    for m in m_list:
        # print(m)
        m_times = []
        for sample in samples:
            # print('\t', sample)
            s_dir = osp.join(dir, str(m), 'samples', f'sample{sample}')
            for f in os.listdir(s_dir):
                if f.startswith('optimize_grid') and f.endswith('max_ent10'):
                    max_ent_dir = osp.join(s_dir, f)
                    break
            else:
                raise Exception(f'Max ent not found for {s_dir}')
            times = []
            for it in range(16):
                f = osp.join(max_ent_dir, f'iteration{it}')
                if osp.exists(f):
                    times.append(load_time_dir(f))
            tot_time = np.sum(times) / 60 # to mins
            times_dict[m].append(tot_time)
            # print('\t', times)
            gnn_times_dict[m].append(times[-1] / 60)

    means = []
    stds = []
    gnn_means = []
    gnn_stds = []
    for m in m_list:
        times = times_dict[m]
        mean = np.mean(times)
        std = np.std(times)
        means.append(mean)
        stds.append(std)

        gnn_times = gnn_times_dict[m]
        gnn_mean = np.mean(gnn_times)
        gnn_std = np.std(gnn_times)
        gnn_means.append(gnn_mean)
        gnn_stds.append(gnn_std)

    # to array
    means = np.array(means)
    gnn_means = np.array(gnn_means)
    ratio = means / gnn_means
    print(means)
    print(gnn_means)
    print(ratio)


    label_fontsize=16
    tick_fontsize=14
    fig, ax = plt.subplots()
    ax.errorbar(m_list, means, stds, color = 'b', label = 'Maximum Entropy')
    ax.errorbar(m_list, gnn_means, gnn_stds, color = 'r', label = 'GNN')

    ax2 = ax.twinx()
    ax2.plot(m_list, ratio, color = 'k')
    ax2.set_ylabel('Ratio', fontsize=label_fontsize)

    ax.set_ylabel('Time (mins)', fontsize=label_fontsize)
    ax.set_xlabel('Number of particles (m)', fontsize=label_fontsize)
    ax.legend(loc='upper left', fontsize=label_fontsize)
    ax.set_xticks(m_list)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig('/home/erschultz/TICG-chromatin/figures/timing.png')





if __name__ == '__main__':
    main()
