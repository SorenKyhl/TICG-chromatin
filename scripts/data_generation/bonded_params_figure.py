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
from pylib.utils.utils import load_import_log
from pylib.utils.xyz import xyz_load, xyz_to_distance

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder

sys.path.append('/home/erschultz/TICG-chromating')
from scripts.distances_Su2020.utils import calculate_rg


def get_bonded_dir(ar, m, b, v, phi, angle):
    root = '/home/erschultz/dataset_bonded'
    if ar == 1.0:
        boundary = 'boundary_spherical'
    else:
        boundary = f'boundary_spheroid_{ar}'

    if v is None:
        volume = f'phi_{phi}'
    else:
        assert phi is None
        volume = f'v_{v}'

    return osp.join(root, boundary, 'bond_type_gaussian', f'm_{m}',
                    f'bond_length_{b}', volume, f'angle_{angle}')

def main(use_max_ent=True, use_v=False):
    label_fontsize=24
    tick_fontsize=18
    letter_fontsize=26

    s = '1013_rescale1'
    s_dir = f'/home/erschultz/Su2020/samples/sample{s}'
    m=512
    log_labels = np.linspace(0, 50000*(m-1), m)
    odir = '/home/erschultz/TICG-chromatin/figures'

    D_exp = np.load(f'/home/erschultz/Su2020/samples/sample{s}/D_crop.npy')
    meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
    D_exp2 = np.load('/home/erschultz/Su2020/samples/sample1/dist2_mean.npy')
    meanDist_D_exp2 = DiagonalPreprocessing.genomic_distance_statistics(D_exp2, mode='freq')
    m2 = len(meanDist_D_exp2)

    def plot_meanDist_D(ax, log, c, dir, label, ls='solid', throw_exception=False):
        if osp.exists(dir):
            # print(dir)
            if use_max_ent:
                final = get_final_max_ent_folder(dir)
                xyz_file = osp.join(final, 'production_out/output.xyz')
            else:
                xyz_file = osp.join(dir, 'production_out/output.xyz')
            xyz = xyz_load(xyz_file, multiple_timesteps=True, verbose=False)
            D = xyz_to_distance(xyz)
            D = np.nanmean(D, axis = 0)
            meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
            ax.plot(log_labels[1:], meanDist_D[1:], label = label, c = c, ls=ls)
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

        v = 8; phi = None; ar=1.5; angle=0
        cmap = mpl.colormaps["Greens"]
        colors = [cmap(0.4), cmap(.6), cmap(.8), cmap(1.0)]
        ls_list = ['solid', 'solid', 'dashed', 'solid']
        for b, c, ls in zip([160, 180, 200, 220], colors, ls_list):
            if use_max_ent:
                if use_v:
                    dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
                else:
                    dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
                if ar != 1.0:
                    dir += f'_spheroid_{ar}'
                dir += '-max_ent10'
            else:
                dir = get_bonded_dir(ar, m, b, v, phi, angle)
            plot_meanDist_D(axes[0], log, c, dir, b, ls)

        b = 200
        cmap = mpl.colormaps["Purples"]
        if use_v:
            colors = [cmap(0.4), cmap(0.6), cmap(0.8), cmap(1.0)]
            ls_list = ['solid', 'dashed', 'solid', 'solid']
            for v, c, ls in zip([6, 8, 10, 12], colors, ls_list):
                if use_max_ent:
                    dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
                    if ar != 1.0:
                        dir += f'_spheroid_{ar}'
                    dir += '-max_ent10'
                else:
                    dir = get_bonded_dir(ar, m, b, v, phi, angle)
                plot_meanDist_D(axes[1], log, c, dir, v, ls)
        else:
            colors = [cmap(0.4), cmap(0.6), cmap(0.8), 'blue', cmap(1.0)]
            for phi, c in zip([0.005, 0.006, 0.007, 0.008, 0.009], colors):
                if use_max_ent:
                    dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
                    if ar != 1.0:
                        dir += f'_spheroid_{ar}'
                    dir += '-max_ent10'
                else:
                    dir = osp.join(bonded_dir, f'bond_length_{b}/phi_{phi}/angle_0')
                plot_meanDist_D(axes[1], log, c, dir, phi, ls)

        b = 200; v = 8; phi=None
        cmap = mpl.colormaps["Oranges"]
        colors = [ cmap(0.4), cmap(0.7), cmap(1.0)]
        ls_list = ['solid', 'dashed', 'solid']
        for ar, c, ls in zip([1, 1.5, 2.0], colors, ls_list):
            if use_max_ent:
                if use_v:
                    dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
                else:
                    dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
                if ar != 1.0:
                    dir += f'_spheroid_{ar}'
                dir += '-max_ent10'
            else:
                dir = get_bonded_dir(ar, m, b, v, phi, angle)

            plot_meanDist_D(axes[2], log, c, dir, ar, ls)

    for ax in all_axes.flatten():
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        nan_rows = np.isnan(meanDist_D_exp)
        ax.plot(log_labels[~nan_rows][1:], meanDist_D_exp[~nan_rows][1:],
                    label='', color='k')
        ax.set_ylim([0, None])
        ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500])

        nan_rows = np.isnan(meanDist_D_exp2)
        ax.plot(np.linspace(0, 30000*(m2-1), m2)[~nan_rows][1:],
                    meanDist_D_exp2[~nan_rows][1:], label='', color='k', ls=':')

    for n, ax in enumerate(all_axes[:,0]):
        inds = [0, m/4, m/2, 3*m/4, m-1]
        inds = np.array(inds).astype(int)
        ticks = [log_labels[i] for i in inds]
        labels = np.array(ticks) / 1000 / 1000 # to mb
        labels = np.round(labels, 0).astype(int)
        ax.set_xticks(ticks, labels = labels)
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                    size=letter_fontsize, weight='bold')

    for ax in all_axes[:,1]:
        ax.set_xlim([50000, None])

    if use_v:
        labels = [r'Bond Length, $b$', r'Volume, $V$', 'Aspect Ratio']
    else:
        labels = [r'Bond Length, $b$', r'Volume Fraction, $\bar{\phi}$', 'Aspect Ratio']
    for ax, label in zip(all_axes[:, 1], labels):
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
    if use_max_ent:
        plt.savefig(osp.join(odir, 'd_s_max_ent.png'))
    else:
        plt.savefig(osp.join(odir, 'd_s_bonded.png'))
    plt.close()

def rg_figure(use_max_ent=True, use_v=False):
    label_fontsize=24
    tick_fontsize=18
    letter_fontsize=26

    if use_max_ent:
        m=512
        s = '1013_rescale1'
        s_dir = f'/home/erschultz/Su2020/samples/sample{s}'
        result = load_import_log(s_dir)
        resolution = result['resolution']
        chrom = int(result['chrom'])
    else:
        m=512
        chrom=21
        resolution=50000

    sizes = [16, 32, 64, 128, 256, 512]
    log_labels = [i*resolution for i in sizes]
    odir = '/home/erschultz/TICG-chromatin/figures'

    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz_exp = np.load(xyz_file)
    print(xyz_exp.shape)
    xyz2_file = osp.join('/home/erschultz/Su2020/samples/sample1', 'xyz2.npy')
    xyz2_exp = np.load(xyz2_file)
    print(xyz2_exp.shape)

    def plot_Rg(ax, c, dir, label, ls='solid', throw_exception=False):
        if osp.exists(dir):
            # print(dir)
            if use_max_ent:
                final = get_final_max_ent_folder(dir)
                xyz_file = osp.join(final, 'production_out/output.xyz')
            else:
                xyz_file = osp.join(dir, 'production_out/output.xyz')
            xyz = xyz_load(xyz_file, multiple_timesteps=True, verbose=False)

            rg_arr = np.zeros((len(sizes), 2))
            for i, size in enumerate(sizes):
                left = int(256 - size/2)
                right = int(256 + size/2)
                xyz_size = xyz[:, left:right, :]

                rg_arr[i] = calculate_rg(xyz_size, verbose = False)

            ax.errorbar(log_labels, rg_arr[:, 0], rg_arr[:, 1], color = c, ls = ls,
                        label = label)
        elif throw_exception:
            raise Exception(f'{dir} does not exist')
        else:
            print(f'{dir} does not exist')

    print('Starting Figure')
    rows = 4
    fig, axes = plt.subplots(rows, 1)
    print(axes.shape)
    fig.set_figheight(rows*4)
    fig.set_figwidth(10)

    v = 8; phi = None; ar=1.5; angle=0
    cmap = mpl.colormaps["Greens"]
    colors = [cmap(0.4), cmap(.6), cmap(.8), cmap(1.0)]
    ls_list = ['solid', 'solid', 'dashed', 'solid']
    for b, c, ls in zip([120, 180, 200, 220], colors, ls_list):
        if use_max_ent:
            if use_v:
                dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
            else:
                dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
            if ar != 1.0:
                dir += f'_spheroid_{ar}'
            dir += '-max_ent10'
        else:
            dir = get_bonded_dir(ar, m, b, v, phi, angle)
        plot_Rg(axes[0], c, dir, b, ls)

    b = 200
    cmap = mpl.colormaps["Purples"]
    if use_v:
        colors = [cmap(0.4), cmap(0.6), cmap(0.8), cmap(1.0)]
        ls_list = ['solid', 'dashed', 'solid', 'solid']
        for v, c, ls in zip([6, 8, 10, 12], colors, ls_list):
            if use_max_ent:
                dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
                if ar != 1.0:
                    dir += f'_spheroid_{ar}'
                dir += '-max_ent10'
            else:
                dir = get_bonded_dir(ar, m, b, v, phi, angle)
            plot_Rg(axes[1], c, dir, v, ls)
    else:
        colors = [cmap(0.4), cmap(0.6), cmap(0.8), 'blue', cmap(1.0)]
        for phi, c in zip([0.005, 0.006, 0.007, 0.008, 0.009], colors):
            if use_max_ent:
                dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
                if ar != 1.0:
                    dir += f'_spheroid_{ar}'
                dir += '-max_ent10'
            else:
                dir = get_bonded_dir(ar, m, b, v, phi, angle)
            plot_Rg(axes[1], c, dir, phi, ls)

    b = 200; v = 8
    cmap = mpl.colormaps["Oranges"]
    colors = [ cmap(0.4), cmap(0.7), cmap(1.0)]
    ls_list = ['solid', 'dashed', 'solid']
    for ar, c, ls in zip([1, 1.5, 2.0], colors, ls_list):
        if use_max_ent:
            if use_v:
                dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
            else:
                dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
            if ar != 1.0:
                dir += f'_spheroid_{ar}'
            dir += '-max_ent10'
        else:
            dir = get_bonded_dir(ar, m, b, v, phi, angle)

        plot_Rg(axes[2], c, dir, ar, ls)


    b = 200; v = 8; ar=1.5
    cmap = mpl.colormaps["Blues"]
    colors = [cmap(0.4), cmap(.6), cmap(.8), cmap(1.0)]
    ls_list = ['dashed', 'solid', 'solid', 'solid']
    for angle, c, ls in zip([0, 0.1, 1, 10], colors, ls_list):
        if use_max_ent:
            pass
        else:
            dir = get_bonded_dir(ar, m, b, v, phi, angle)

        plot_Rg(axes[3], c, dir, angle, ls)


    for n, ax in enumerate(axes):
        # add experimental data
        sizes = [16, 32, 64, 128, 256, 512]
        rg_arr = np.zeros((len(sizes), 2))
        N, m, _ = xyz_exp.shape
        for i, m in enumerate([1000, 2000, xyz_exp.shape[1]//2]):
            log_labels = [i*50000 for i in sizes]
            for i, size in enumerate(sizes):
                left = int(m - size/2)
                right = int(m + size/2)
                xyz_size = xyz_exp[:, left:right, :]

                rg_arr[i] = calculate_rg(xyz_size, verbose = False)

            ax.errorbar(log_labels, rg_arr[:, 0], rg_arr[:, 1], color = 'k',
                        label = f'Experiment1.{i}')

        sizes = [16, 32, 64]
        rg_arr = np.zeros((len(sizes), 2))
        N, m, _ = xyz2_exp.shape
        log_labels = [i*30000 for i in sizes]
        for i, size in enumerate(sizes):
            left = int(m//2 - size/2)
            right = int(m//2 + size/2)
            xyz_size = xyz2_exp[:, left:right, :]

            rg_arr[i] = calculate_rg(xyz_size, verbose = False)

        ax.errorbar(log_labels, rg_arr[:, 0], rg_arr[:, 1], color = 'k', ls = ':',
                    label = 'Experiment2')

        # set tick size
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # add bold letters
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                    size=letter_fontsize, weight='bold')

        # log scale
        ax.set_xscale('log')


    if use_v:
        labels = [r'Bond Length, $b$', r'Volume, $V$', 'Aspect Ratio', 'K Angle']
    else:
        labels = [r'Bond Length, $b$', r'Volume Fraction, $\bar{\phi}$',
                    'Aspect Ratio']
    for ax, label in zip(axes, labels):
        ax.legend(title = label, fontsize = tick_fontsize,
                    title_fontsize = label_fontsize,
                    bbox_to_anchor=(1, 0.5), loc="center left")

    for ax in axes[:2].flatten():
        ax.set_xticklabels([])

    fig.supylabel('Radius of Gyration', fontsize=label_fontsize)
    axes[2].set_xlabel('Genomic Separation (Mb)', fontsize=label_fontsize)

    plt.tight_layout()
    if use_max_ent:
        plt.savefig(osp.join(odir, 'rg_max_ent.png'))
    else:
        plt.savefig(osp.join(odir, 'rg_bonded.png'))
    plt.close()

if __name__ == '__main__':
    # main(False, True)
    rg_figure(False, True)
