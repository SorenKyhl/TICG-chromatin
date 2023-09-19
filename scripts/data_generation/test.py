import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from modify_maxent import get_samples
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_diag_chi_step
from pylib.utils.plotting_utils import BLUE_RED_CMAP, plot_mean_dist
from pylib.utils.xyz import xyz_load, xyz_to_contact_grid, xyz_to_distance
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_max_ent_S, load_S, load_Y)


def split_samples(dataset):
    samples, _ = get_samples(dataset)
    good = []; bad = []
    for s in samples:
        s_dir = osp.join('/home/erschultz', dataset, f'samples/sample{s}')
        y, _ = load_Y(s_dir)
        y /= np.mean(np.diagonal(y))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
        if meanDist[10] > 0.06:
            bad.append(s)
        else:
            good.append(s)

    return good, bad

def compare_diag_params():
    dataset = 'dataset_08_24_23_v4'
    data_dir = osp.join('/home/erschultz', dataset)
    good, bad = split_samples(dataset)
    print(f'good = {len(good)}, bad = {len(bad)}')

    def get_data(s):
        s_dir = osp.join(data_dir, f'samples/sample{s}')
        diag_chis = np.load(osp.join(s_dir, 'diag_chis.npy'))
        chis = np.load(osp.join(s_dir, 'chis.npy'))
        chis = chis.diagonal()
        with open(osp.join(s_dir, 'config.json')) as f:
            config = json.load(f)
        setup_file = osp.join(data_dir, f'setup/sample_{s}.txt')
        with open(setup_file) as f:
            for line in f:
                if line.startswith('--diag_chi_experiment'):
                    line = f.readline()
                    exp = line.strip()
                elif line.startswith('--exp_max_ent'):
                    line = f.readline()
                    exp_j = line.strip()
                    exp = osp.join('dataset_02_04_23/samples', f'sample{exp_j}', 'optimize_grid_b_261_phi_0.01')

        exp_chis = np.load(osp.join('/home/erschultz', exp+'-max_ent10/chis_eig_norm.npy'))
        exp_chis = exp_chis.diagonal()
        return diag_chis, chis, exp_chis, config

    for c, data in zip(['g', 'r'], [good, bad]):
        for i in data:
            diag_chis, chis, exp_chis, config = get_data(i)
            plt.plot(chis, c=c)

    plt.xlabel('Chi')
    plt.ylabel('Value')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/bad_vs_good_chis.png')
    plt.close()

    for c, data in zip(['g', 'r'], [good, bad]):
        for i in data:
            diag_chis, chis, exp_chis, config = get_data(i)
            plt.plot(diag_chis, c=c)

    plt.xlabel('Diag Chi')
    plt.ylabel('Value')
    plt.xscale('log')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/bad_vs_good_diag_chis.png')
    plt.close()

def compare_S(dataset):
    data_dir = osp.join('/home/erschultz', dataset)
    samples, _ = get_samples(dataset)
    samples = samples[:10]
    GNN_ID=434
    b=140; phi=0.03

    for s in samples:
        s_dir = osp.join(data_dir, f'samples/sample{s}')
        max_ent_dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}-max_ent10')
        final = get_final_max_ent_folder(max_ent_dir)
        S_max_ent = load_max_ent_S(max_ent_dir)
        meanDist_S_max_ent = DiagonalPreprocessing.genomic_distance_statistics(S_max_ent, mode='freq')
        plot_mean_dist(meanDist_S_max_ent, final, 'meanDist_S_log.png', None,
                        logx = True, logy = False,
                        ylabel = 'mean(diagonal(S, d))')

        gnn_dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}-GNN{GNN_ID}')
        S_gnn = np.load(osp.join(gnn_dir, 'S.npy'))
        meanDist_S_gnn = DiagonalPreprocessing.genomic_distance_statistics(S_gnn, mode='freq')

        S = np.load(osp.join(s_dir, 'S.npy'))
        meanDist_S = DiagonalPreprocessing.genomic_distance_statistics(S, mode='freq')

        plot_mean_dist(meanDist_S, s_dir, 'meanDist_S_comparison.png', None,
                        logx = True, logy = False,
                        ref = meanDist_S_gnn, ref_label = 'GNN', ref_color='red',
                        ref2 = meanDist_S_max_ent, ref2_label = 'Max Ent', ref2_color = 'blue',
                        label = 'Reference', color = 'k',
                        ylabel = 'mean(diagonal(S, d))')


        fig, (ax1, ax2, ax3, axcb) = plt.subplots(1, 4,
                                gridspec_kw={'width_ratios':[1,1,1,0.08]})
        fig.set_figheight(6)
        fig.set_figwidth(6*2.5)
        fig.suptitle(f'Sample {s}', fontsize = 16)
        arr = np.array([S, S_max_ent])
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        vmax = max(vmax, vmin * -1)
        vmin = vmax * -1
        s1 = sns.heatmap(S, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                        ax = ax1, cbar = False)
        s1.set_title(f'$S$', fontsize = 16)
        s1.set_yticks([])
        s2 = sns.heatmap(S_max_ent, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                        ax = ax2, cbar = False)
        s2.set_title(r'Max Ent $\hat{S}$', fontsize = 16)
        s2.set_yticks([])
        s3 = sns.heatmap(S - S_max_ent, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                        ax = ax3, cbar_ax = axcb)
        title = ('Difference\n'
                r'($S$ - Max Ent $\hat{S}$)')
        s3.set_title(title, fontsize = 16)
        s3.set_yticks([])

        plt.tight_layout()
        plt.savefig(osp.join(s_dir, 'S_vs_max_ent_S.png'))
        plt.close()

def compare_S2():
    S1 = np.load('/home/erschultz/dataset_02_04_23/samples/sample204/optimize_grid_b_261_phi_0.01-max_ent10/iteration30/S.npy')
    S2 = np.load('/home/erschultz/dataset_02_04_23/samples/sample204/optimize_grid_b_261_phi_0.01-max_ent10-init_diag/iteration30/S.npy')


    fig, (ax1, ax2, ax3, axcb) = plt.subplots(1, 4,
                            gridspec_kw={'width_ratios':[1,1,1,0.08]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    arr = np.array([S1, S2])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    s1 = sns.heatmap(S1, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax1, cbar = False)
    s1.set_title(f'$S1$', fontsize = 16)
    s1.set_yticks([])
    s2 = sns.heatmap(S2, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax2, cbar = False)
    s2.set_title(r'$S2$', fontsize = 16)
    s2.set_yticks([])
    diff = S1 - S2
    rmse = mean_squared_error(S1, S2, squared=False)
    s3 = sns.heatmap(diff, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax3, cbar_ax = axcb)
    title = (f'Difference (RMSE: {rmse:.2f})\n'
            r'($S1$ - $S2$)')
    s3.set_title(title, fontsize = 16)
    s3.set_yticks([])

    plt.tight_layout()
    plt.savefig(osp.join('/home/erschultz/dataset_02_04_23/samples/sample204/', 'S_vs_S.png'))
    plt.close()

def compare_p_s_bonded():
    '''Compare c++ to python implementation'''
    dir = '/home/erschultz/dataset_bonded/bond_type_DSS/m_512/bond_length_140/phi_0.01'
    y = np.loadtxt(osp.join(dir, 'production_out/contacts.txt'))
    y = y.astype(float)
    y /= np.mean(np.diagonal(y))
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    plt.plot(meanDist, label = 'c++')


    xyz_file = osp.join(dir, 'production_out/output.xyz')
    xyz = xyz_load(xyz_file, multiple_timesteps=True)
    gs=100
    y = xyz_to_contact_grid(xyz, gs, dtype=float)
    y /= np.mean(np.diagonal(y))
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    plt.plot(meanDist, label = 'python')


    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('s')
    plt.ylabel(rf'P(s) calculated ex post facto w/ $\Delta=${gs}')
    plt.legend()
    plt.savefig(osp.join(dir, 'p_s.png'))
    plt.close()

def compare_p_s_bonded2():
    '''Compare p(s) curves in dataset_bonded'''
    dataset = '/home/erschultz/dataset_bonded'
    m=1024; phi=0.06; b=140; bond_type='gaussian'
    for boundary_type, ar, c in [('spheroid', 2.0, 'g')]:
        if boundary_type == 'spheroid':
            boundary_type += f'_{ar}'
        for k_angle in [0]:
            dir = osp.join(dataset, f'boundary_{boundary_type}',
                        f'bond_type_{bond_type}/m_{m}/bond_length_{b}/phi_{phi}')
            if k_angle != 0:
                dir = osp.join(dir, f'angle_{k_angle}')
            xyz_file = osp.join(dir, 'production_out/output.xyz')
            xyz = xyz_load(xyz_file, multiple_timesteps=True)

            gs=300
            y = xyz_to_contact_grid(xyz, gs, dtype=float)
            y /= np.mean(np.diagonal(y))
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
            plt.plot(meanDist, ls = ls, c = c, label = f'ar_{ar}_k_angle_{k_angle}')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('s')
    plt.ylabel(rf'P(s) w/ $\Delta=${gs} nm')
    plt.legend(title='bonded params')
    plt.savefig(osp.join(dataset, 'p_s_bonded.png'))
    plt.close()

def compare_d_s_bonded():
    '''Compare d(s) curves in dataset_bonded'''
    dataset = '/home/erschultz/dataset_bonded'
    m=512; phi=0.01; b=261; bond_type='gaussian'
    ls='-'
    for boundary_type, ar in [('spherical', 1.0)]:
        if boundary_type == 'spheroid':
            boundary_type += f'_{ar}'
        for b in [140, 261]:
            for k_angle in np.arange(0.1, 1, .2):
                k_angle = np.round(k_angle, 1)
                dir = osp.join(dataset, f'boundary_{boundary_type}',
                            f'bond_type_{bond_type}/m_{m}/bond_length_{b}/phi_{phi}')
                if k_angle != 0:
                    dir = osp.join(dir, f'angle_{k_angle}')
                print(dir)
                xyz_file = osp.join(dir, 'production_out/output.xyz')
                xyz = xyz_load(xyz_file, multiple_timesteps=True)
                D = xyz_to_distance(xyz)
                log_labels = np.linspace(0, (m-1), m)
                D = np.nanmean(D, axis = 0)
                meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
                plt.plot(log_labels, meanDist_D, ls = ls, label = f'ar_{ar}_k_angle_{k_angle}')

    D_exp = np.load('/home/erschultz/Su2020/samples/sample1013/D_crop.npy')
    meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
    nan_rows = np.isnan(meanDist_D_exp)
    plt.plot(log_labels[~nan_rows], meanDist_D_exp[~nan_rows],
                label='Experiment', color='k')

    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Distance (nm)')
    plt.xlabel('Polymer Distance (m)')
    plt.legend(title='bonded params')
    plt.savefig(osp.join(osp.join(dataset, f'boundary_{boundary_type}',
                f'bond_type_{bond_type}/m_{m}/bond_length_{b}/phi_{phi}'), 'd_s_bonded.png'))
    plt.close()

def compare_d_s_bonded2():
    '''Compare d(s) curves in dataset_bonded'''
    m=512
    log_labels = np.linspace(0, (m-1), m)
    def plot_meanDist_D(dir, label):
        xyz_file = osp.join(dir, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps=True)
        D = xyz_to_distance(xyz)
        D = np.nanmean(D, axis = 0)
        meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
        plt.plot(log_labels, meanDist_D, label = label)

    plot_meanDist_D('/home/erschultz/dataset_bonded/boundary_spherical/bond_type_gaussian/m_512/bond_length_261/phi_0.01',
                    label='Gaussian')
    plot_meanDist_D('/home/erschultz/dataset_bonded/boundary_spherical/bond_type_FENE/m_512/bond_length_261/phi_0.01/angle_2',
                    label='FENE+cosine')

    D_exp = np.load('/home/erschultz/Su2020/samples/sample1013/D_crop.npy')
    meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
    nan_rows = np.isnan(meanDist_D_exp)
    plt.plot(log_labels[~nan_rows], meanDist_D_exp[~nan_rows],
                label='Experiment', color='k')

    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Distance (nm)')
    plt.xlabel('Polymer Distance (m)')
    plt.legend(title='bonded params')
    plt.savefig(osp.join('/home/erschultz/TICG-chromatin/figures', 'd_s_bonded2.png'))
    plt.close()



def compare_p_s_bonded3():
    '''Compare p(s) after optimizing grid size for specific experiment'''
    data_dir = '/home/erschultz/Su2020/samples/sample1013'
    b=261; phi=0.01
    for b in [180, 200, 220, 240]:
        for phi in [0.01, 0.02, 0.03, 0.04]:
            for ar in [1.0]:
                if ar == 1:
                    dir = osp.join(data_dir, f'optimize_grid_b_{b}_phi_{phi}')
                else:
                    dir = osp.join(data_dir, f'optimize_grid_b_{b}_phi_{phi}_spheroid_{ar}')
                y = np.load(osp.join(dir, 'y.npy'))
                y = y.astype(float)
                y /= np.mean(np.diagonal(y))
                meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
                meanDist /= meanDist[1]
                plt.plot(meanDist, label = f'b_{b}_phi_{phi}_ar_{ar}')

    y = np.load(osp.join(data_dir, 'y.npy'))
    y = y.astype(float)
    y /= np.mean(np.diagonal(y))
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    meanDist /= meanDist[1]
    plt.plot(meanDist, label = 'Experiment', c='k')


    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('s')
    plt.ylabel('P(s)')
    plt.legend()
    plt.savefig(osp.join(data_dir, 'p_s_bonded.png'))
    plt.close()

def compare_d_s_max_ent():
    '''Compare p(s) curves in dataset_bonded'''
    s_dir = '/home/erschultz/Su2020/samples/sample1013'
    m=512
    log_labels = np.linspace(0, 50000*(m-1), m)
    def plot_meanDist_D(dir, label):
        final = get_final_max_ent_folder(dir)
        xyz_file = osp.join(final, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps=True)
        D = xyz_to_distance(xyz)
        D = np.nanmean(D, axis = 0)
        meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
        plt.plot(log_labels, meanDist_D, label = label)

    D_exp = np.load('/home/erschultz/Su2020/samples/sample1013/D_crop.npy')
    meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
    D_exp2 = np.load('/home/erschultz/Su2020/samples/sample1/dist2_mean.npy')
    meanDist_D_exp2 = DiagonalPreprocessing.genomic_distance_statistics(D_exp2, mode='freq')
    print(meanDist_D_exp2)
    m2 = len(meanDist_D_exp2)

    for log in [True, False]:
        for b in [180, 200, 220, 240, 261]:
            for phi in [0.01]:
                plot_meanDist_D(osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}-max_ent10'),
                                f'b_{b}_phi_{phi}')

        nan_rows = np.isnan(meanDist_D_exp)
        plt.plot(log_labels[~nan_rows], meanDist_D_exp[~nan_rows],
                    label='Experiment', color='k')

        # nan_rows = np.isnan(meanDist_D_exp2)
        # plt.plot(np.linspace(0, 30000*(m2-1), m2)[~nan_rows], meanDist_D_exp2[~nan_rows],
        #             label='Experiment 2', color='k', ls=':')

        plt.ylabel('Distance (nm)', fontsize=16)
        plt.xlabel('Genomic Separation (bp)', fontsize=16)
        plt.legend()
        if log:
            plt.xscale('log')
            plt.savefig(osp.join(s_dir, 'd_s_max_ent_log.png'))
        else:
            plt.savefig(osp.join(s_dir, 'd_s_max_ent.png'))
        plt.close()


def compare_meanDist_S():
    sample = 2
    data_dir = '/home/erschultz/dataset_09_17_23/samples'
    s_dir = osp.join(data_dir, f'sample{sample}')
    Sgnn = np.load(osp.join(s_dir, 'optimize_grid_b_261_phi_0.01-GNN457/S.npy'))
    Sgt = np.load(osp.join(s_dir, 'S.npy'))
    for S, label in zip([Sgt, Sgnn],['Ground Truth', 'GNN']):
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
        plt.plot(meanDist, label = label)
    plt.legend()
    plt.xscale('log')
    plt.ylabel('Mean', fontsize=16)
    plt.xlabel('Off-diagonal Index', fontsize=16)
    plt.savefig(osp.join(s_dir, 'meanDist_S.png'))
    plt.close()

def compare_p_s_modified():
    dataset='dataset_02_04_23'
    data_dir = osp.join('/home/erschultz', dataset)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for sample in range(201, 211):
        s_dir = osp.join(data_dir, 'samples', f'sample{sample}')
        y_exp = np.load(osp.join(s_dir, 'y.npy'))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob')
        ax.plot(meanDist, c = 'k')

        b=261; phi=0.01; k=10
        max_ent_dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}-max_ent{k}')
        for f in ['copy_S_delta']:
            f_dir = osp.join(max_ent_dir, f)
            y = np.load(osp.join(f_dir, 'y.npy'))
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            ax.plot(meanDist, label = sample)

            S = np.load(osp.join(f_dir, 'S.npy'))
            meanDist_S = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
            ax2.plot(meanDist_S)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Genomic Distance (s)')
    ax.set_ylabel('P(s)')
    ax.set_ylabel(r'Mean$(S_{|i-j|=s}$ ')
    ax.legend()
    plt.tight_layout()
    plt.savefig(osp.join(data_dir, 'meanDist_modified.png'))
    plt.close()

def check_GNN_S():
    e = np.loadtxt('/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/434/dataset_04_28_23_sample324/sample324/energy.txt')
    e_orig = np.multiply(np.sign(e), np.exp(np.abs(e)) - 1)
    print(e_orig[1, :10], e_orig.shape)

    S = load_S('/home/erschultz/dataset_04_28_23/samples/sample324')
    # S = np.load('/home/erschultz/dataset_04_28_23/samples/sample324/S.npy')
    print(S[1, :10])

def compare_xyz():
    dir1 = '/home/erschultz/Su2020/samples/sample1004/optimize_grid_b_261_phi_0.01 (copy)/iteration6'
    dir2 = '/home/erschultz/dataset_bonded/m_512/bond_length_261/phi_0.01/production_out'
    dir3 = '/home/erschultz/dataset_bonded/m_512/bond_length_261/phi_0.01/equilibration'

    for i, dir in enumerate([dir1, dir2]):
        energy = np.loadtxt(osp.join(dir, 'energy.traj'))[:, 1]
        avg = np.mean(energy)
    #     plt.plot(energy, label=f'{i}, mean={avg}')
    # plt.legend()
    # plt.show()

    config_file1 = osp.join(dir1, 'config.json')
    with open(config_file1) as f:
        config1 = json.load(f)
    config_file2 = osp.join(dir2, 'config.json')
    with open(config_file2) as f:
        config2 = json.load(f)
    for k, val1 in config1.items():
        val2 = config2[k]
        if val1 != val2:
            print(k, val1, val2)


if __name__ == '__main__':
    # compare_diag_params()
    # compare_S('dataset_04_28_23')
    # compare_S2()
    # compare_p_s_bonded3()
    # compare_d_s_bonded()
    # compare_d_s_bonded2()
    # compare_d_s_max_ent()
    # compare_meanDist_S()
    # compare_p_s_modified()
    # compare_xyz()
    check_GNN_S()
