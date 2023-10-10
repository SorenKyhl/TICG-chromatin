import json
import os
import os.path as osp
import string
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from modify_maxent import get_samples
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.xyz import xyz_load, xyz_to_distance

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder


def main():
    label_fontsize=24
    tick_fontsize=18
    letter_fontsize=26


    s = 1013
    s_dir = f'/home/erschultz/Su2020/samples/sample{s}'
    m=512
    log_labels = np.linspace(0, 50000*(m-1), m)
    odir = '/home/erschultz/TICG-chromatin/figures'

    D_exp = np.load(f'/home/erschultz/Su2020/samples/sample{s}/D_crop.npy')
    meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
    # D_exp2 = np.load('/home/erschultz/Su2020/samples/sample1/dist2_mean.npy')
    # meanDist_D_exp2 = DiagonalPreprocessing.genomic_distance_statistics(D_exp2, mode='freq')
    # m2 = len(meanDist_D_exp2)

    def plot_meanDist_D(ax, log, c, dir, label, throw_exception=False):
        if osp.exists(dir):
            final = get_final_max_ent_folder(dir)
            xyz_file = osp.join(final, 'production_out/output.xyz')
            xyz = xyz_load(xyz_file, multiple_timesteps=True, verbose=False)
            D = xyz_to_distance(xyz)
            D = np.nanmean(D, axis = 0)
            meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
            ax.plot(log_labels[1:], meanDist_D[1:], label = label, c = c)
            if log:
                ax.set_xscale('log')
        elif throw_exception:
            raise Exception(f'{dir} does not exist')
        else:
            print(f'{dir} does not exist')


    print('Starting Figure')
    fig, all_axes = plt.subplots(3, 2)
    print(all_axes.shape)
    fig.set_figheight(12)
    fig.set_figwidth(16)

    for log in [True, False]:
        if log:
            axes = all_axes[:,1]
        else:
            axes = all_axes[:, 0]

        phi = 0.008
        cmap = mpl.colormaps["Greens"]
        colors = [cmap(0.1), 'blue', cmap(.4), cmap(0.7), cmap(1.0)]
        for b, c in zip([160, 180, 200, 220, 240], colors):
            plot_meanDist_D(axes[0], log, c,
                            osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}_spheroid_1.5-max_ent10'),
                            b)

        b = 180
        cmap = mpl.colormaps["Purples"]
        colors = [cmap(0.1), cmap(0.4), 'blue', cmap(0.7), cmap(1.0)]
        for phi, c in zip([0.004, 0.006, 0.008, 0.01, 0.02], colors):
            plot_meanDist_D(axes[1], log, c,
                            osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}_spheroid_1.5-max_ent10'),
                            phi)

        b = 180; phi = 0.008
        cmap = mpl.colormaps["Oranges"]
        colors = [ cmap(0.4), 'blue', cmap(1.0)]
        for ar, c in zip([1, 1.5, 2.0], colors):
            if ar == 1:
                dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}-max_ent10')
            else:
                dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}_spheroid_{ar}-max_ent10')
            plot_meanDist_D(axes[2], log, c, dir, ar)

    for axes in all_axes:
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            nan_rows = np.isnan(meanDist_D_exp)
            ax.plot(log_labels[~nan_rows][1:], meanDist_D_exp[~nan_rows][1:],
                        label='', color='k')
            ax.set_yticks([250, 500, 750, 1000, 1250, 1500])

            # nan_rows = np.isnan(meanDist_D_exp2)
            # ax.plot(np.linspace(0, 30000*(m2-1), m2)[~nan_rows][1:], meanDist_D_exp2[~nan_rows][1:],
                        # label='Experiment 2', color='k', ls=':')

    for n, ax in enumerate(all_axes[:,0]):
        inds = [0, m/4, m/2, 3*m/4, m-1]
        inds = np.array(inds).astype(int)
        ticks = [log_labels[i] for i in inds]
        labels = np.array(ticks) / 1000 / 1000 # to mb
        labels = np.round(labels, 0).astype(int)
        ax.set_xticks(ticks, labels = labels)
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                    size=letter_fontsize, weight='bold')

    for ax, label in zip(all_axes[:, 1], [r'Bond Length, $b$', r'Volume Fraction, $\bar{\phi}$', 'Aspect Ratio']):
        ax.legend(title = label, fontsize = tick_fontsize,
                    title_fontsize = label_fontsize,
                    bbox_to_anchor=(1, 0.5), loc="center left")
        ax.set_yticklabels([])
        # ax.set_ylabels([])

    for ax in all_axes[:2, :].flatten():
        ax.set_xticklabels([])

    fig.supylabel('Distance (nm)', fontsize=label_fontsize)
    all_axes[2,0].set_xlabel('Genomic Separation (Mb)', fontsize=label_fontsize)
    all_axes[2,1].set_xlabel('Genomic Separation (bp)', fontsize=label_fontsize)

    plt.tight_layout()
    plt.savefig(osp.join(odir, 'd_s_max_ent.png'))
    plt.close()

if __name__ == '__main__':
    main()
