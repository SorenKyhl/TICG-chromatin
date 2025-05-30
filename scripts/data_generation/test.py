import json
import os
import os.path as osp
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from modify_maxent import get_samples
from pylib.utils import epilib, hic_utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_diag_chi_step
from pylib.utils.plotting_utils import (BLUE_RED_CMAP, RED_CMAP, plot_matrix,
                                        plot_mean_dist)
from pylib.utils.utils import make_composite
from pylib.utils.xyz import xyz_load, xyz_to_contact_grid, xyz_to_distance
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_max_ent_S, load_S, load_Y)


def compare_diag_fits():
    dir = '/home/erschultz/dataset_06_29_23/samples/sample1'
    y_exp = np.load(osp.join(dir, 'y.npy'))
    meanDist_exp = DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob')
    plt.plot(meanDist_exp, c='k', label='exp')

    dir = osp.join(dir, 'optimize_grid_b_180_v_8_spheroid_1.5-max_ent10')
    y_me = np.load(osp.join(dir, 'iteration30/y.npy'))
    meanDist_me = DiagonalPreprocessing.genomic_distance_statistics(y_me, 'prob')
    plt.plot(meanDist_me, label='ME')

    dir = osp.join(dir, 'samples')
    methods = ['S_target']
    for method in methods:
        y = np.load(osp.join(dir, method, 'y.npy'))
        meanDist_y = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist_y, label=method)

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title='Method', fontsize=14,
                bbox_to_anchor=(1, .5), loc='center left')
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('P(S)', fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'p_s_comparison.png'))
    plt.close()

def split_dataset(dataset, s, cutoff):
    print(f'Using s = {s}, cutoff = {cutoff}')
    dir = '/project2/depablo/erschultz'
    # dir = '/home/erschultz'
    odir = osp.join(dir, dataset + f'_s_{s}_cutoff_{cutoff}')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    odir = osp.join(odir, 'samples')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)

    rejects = 0
    for i in range(1, 10001):
        s_dir = osp.join(dir, dataset, f'samples/sample{i}')
        if not osp.exists(s_dir):
            continue
        y, _ = load_Y(s_dir)
        y /= np.mean(np.diagonal(y))
        val = np.nanmean(np.diagonal(y, offset = s))
        if val > cutoff:
            rejects += 1
        else:
            odir_s = osp.join(odir, f'sample{i}')
            if not osp.exists(odir_s):
                os.mkdir(odir_s, mode=0o755)
                os.mkdir(osp.join(odir_s, 'production_out'))
            for j in [200000, 300000, 400000, 500000]:
                shutil.copy(osp.join(s_dir, f'production_out/contacts{j}.txt'),
                            osp.join(odir_s, f'production_out/contacts{j}.txt'))
            shutil.copy(osp.join(s_dir, 'diag_chis.npy'),
                        osp.join(odir_s, 'diag_chis.npy'))
            shutil.copy(osp.join(s_dir, 'L.npy'),
                        osp.join(odir_s, 'L.npy'))

    print(f'Rejected {np.round(rejects / 10000 * 100, 3)} percent')

def split_dataset2(dataset, cutoff):
    print(f'Using cutoff = {cutoff}')
    dir = '/project2/depablo/erschultz'
    # dir = '/home/erschultz'
    odir = osp.join(dir, dataset + f'_cutoff_{cutoff}')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    odir = osp.join(odir, 'samples')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)

    ref_meanDist = np.load(osp.join(dir, 'dataset_02_04_23/molar_contact_ratio/meanDist.npy'))
    ref_meanDist = np.mean(ref_meanDist, axis = 0)

    rejects = 0
    accepts = 0
    for i in range(1, 10001): 
 	s_dir = osp.join(dir, dataset, f'samples/sample{i}')
        if not osp.exists(s_dir):
            continue
        y, _ = load_Y(s_dir)
        y /= np.mean(np.diagonal(y))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, mode='freq')
        val = mean_squared_error(meanDist, ref_meanDist, squared = False)

        if val > cutoff:
            rejects += 1
        else:
            accepts += 1
            odir_s = osp.join(odir, f'sample{i}')
            if osp.exists(odir_s):
                shutil.rmtree(odir_s) # wipe and re-copy to be safe
            os.mkdir(odir_s, mode=0o755)
            shutil.copy(osp.join(s_dir, 'diag_chis.npy'),
                        osp.join(odir_s, 'diag_chis.npy'))
            shutil.copy(osp.join(s_dir, 'L.npy'),
                        osp.join(odir_s, 'L.npy'))
            shutil.copy(osp.join(s_dir, 'y.npy'),
                        osp.join(odir_s, 'y.npy'))

    print(f'Rejected {np.round(rejects / (accepts+rejects) * 100, 3)} percent')


def compare_p_s_exp():
    '''compare different experimantal p_s curves'''
    def plot(file, label):
        y = np.load(file)
        y = y.astype(float)
        y /= np.mean(np.diagonal(y))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
        plt.plot(meanDist, label = label)

    plot('/home/erschultz/Su2020/samples/sample1013/y.npy', 1013)
    plot('/home/erschultz/Su2020/samples/sample1004/y.npy', 1004)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()

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
    m=1024; bond_type='SC'; beadvol=13000; k_bond=0.02
    log_labels = np.linspace(0, 5000*(m-1), m)
    for boundary_type, ar in [('spherical', 1.0)]:
        if boundary_type == 'spheroid':
            boundary_type += f'_{ar}'
        for b in [63, 100]:
            for v in [0.5, 1]:
                for k_angle in [0, 2]:
                    k_angle = np.round(k_angle, 1)
                    dir = osp.join(dataset, f'boundary_{boundary_type}/beadvol_{beadvol}',
                                f'bond_type_{bond_type}')
                    if bond_type == 'SC':
                        dir = osp.join(dir, f'k_bond_{k_bond}')
                    assert osp.exists(dir)
                    dir = osp.join(dir, f'm_{m}/bond_length_{b}/v_{v}/angle_{k_angle}')
                    print(dir)
                    xyz_file = osp.join(dir, 'production_out/output.xyz')
                    xyz = xyz_load(xyz_file, multiple_timesteps=True, N_max=50, verbose=False)
                    D = xyz_to_distance(xyz)
                    D = np.nanmean(D, axis = 0)
                    meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
                    if k_angle == 0:
                        ls = '-'
                    else:
                        ls = ':'
                    plt.plot(log_labels, meanDist_D, ls = ls, label = f'bt_{bond_type}_b_{b}_v_{v}_k_{k_angle}')

    D_exp = np.load(f'/home/erschultz/Su2020/samples/sample1013/D_crop.npy')
    meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
    m1 = len(meanDist_D_exp)
    nan_rows = np.isnan(meanDist_D_exp)
    plt.plot(np.linspace(0, 50000*(m1-1), m1)[~nan_rows][1:m//40],
                meanDist_D_exp[~nan_rows][1:m//40], color='k')

    D_exp2 = np.load('/home/erschultz/Su2020/samples/sample1/dist2_mean.npy')
    meanDist_D_exp2 = DiagonalPreprocessing.genomic_distance_statistics(D_exp2, mode='freq')
    m2 = len(meanDist_D_exp2)
    nan_rows = np.isnan(meanDist_D_exp2)
    plt.plot(np.linspace(0, 30000*(m2-1), m2)[~nan_rows][1:],
                meanDist_D_exp2[~nan_rows][1:], label='Experiment', color='k')

    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Distance (nm)')
    plt.xlabel('Genomic Distance (bp)')
    plt.legend(title='bonded params')
    plt.savefig(osp.join(dataset, 'd_s_bonded.png'))
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
    s=1004
    s_dir = f'/home/erschultz/Su2020/samples/sample{s}'
    odir = osp.join(s_dir, 'd_s_curves')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    m=512
    log_labels = np.linspace(0, 50000*(m-1), m)
    def plot_meanDist_D(dir, label):
        final = get_final_max_ent_folder(dir)
        xyz_file = osp.join(final, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps=True)
        D = xyz_to_distance(xyz)
        D = np.nanmean(D, axis = 0)
        meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
        plt.plot(log_labels[1:], meanDist_D[1:], label = label)

    D_exp = np.load(f'/home/erschultz/Su2020/samples/sample{s}/D_crop.npy')
    meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
    D_exp2 = np.load('/home/erschultz/Su2020/samples/sample1/dist2_mean.npy')
    meanDist_D_exp2 = DiagonalPreprocessing.genomic_distance_statistics(D_exp2, mode='freq')
    m2 = len(meanDist_D_exp2)

    for log in [False, True]:
        if log:
            odir = osp.join(odir, 'log')
            if not osp.exists(odir):
                os.mkdir(odir, mode=0o755)
        for ar in [1.5]:
            for b in [160, 180]:
                for v in [6,  8, 10, 12]:
                    grid_dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
                    if ar != 1.0:
                        grid_dir += f'_spheroid_{ar}'
                    max_ent_dir = grid_dir + '-max_ent10'
                    if osp.exists(max_ent_dir):
                        plot_meanDist_D(max_ent_dir,
                                        f'b_{b}_v_{v}_ar_{ar}')

                # for phi in [0.006, 0.007, 0.008, 0.009, 0.01]:
                #     grid_dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
                #     if ar != 1.0:
                #         grid_dir += f'_spheroid_{ar}'
                #     max_ent_dir = grid_dir + '-max_ent10'
                #     if osp.exists(max_ent_dir):
                #         plot_meanDist_D(max_ent_dir,
                #                         f'b_{b}_phi_{phi}_ar_{ar}')

                nan_rows = np.isnan(meanDist_D_exp)
                plt.plot(log_labels[~nan_rows][1:], meanDist_D_exp[~nan_rows][1:],
                            label='Experiment', color='k')

                nan_rows = np.isnan(meanDist_D_exp2)
                plt.plot(np.linspace(0, 30000*(m2-1), m2)[~nan_rows][1:], meanDist_D_exp2[~nan_rows][1:],
                            label='Experiment 2', color='k', ls=':')

                plt.ylabel('Distance (nm)', fontsize=16)
                plt.xlabel('Genomic Separation (bp)', fontsize=16)
                plt.legend()
                if log:
                    plt.xscale('log')
                    plt.savefig(osp.join(odir, f'd_s_max_ent_b_{b}_ar_{ar}_log.png'))
                else:
                    plt.savefig(osp.join(odir, f'd_s_max_ent_b_{b}_ar_{ar}.png'))
                plt.close()


def compare_d_s_max_ent2():
    '''Compare p(s) curves in dataset_bonded'''
    s=1004
    s_dir = f'/home/erschultz/Su2020/samples/sample{s}'
    odir = osp.join(s_dir, 'd_s_curves')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    m=512
    log_labels = np.linspace(0, 50000*(m-1), m)
    def plot_meanDist_D(dir, label):
        final = get_final_max_ent_folder(dir)
        xyz_file = osp.join(final, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps=True)
        D = xyz_to_distance(xyz)
        D = np.nanmean(D, axis = 0)
        meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
        plt.plot(log_labels[1:], meanDist_D[1:], label = label)

    D_exp = np.load(f'/home/erschultz/Su2020/samples/sample{s}/D_crop.npy')
    meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
    D_exp2 = np.load('/home/erschultz/Su2020/samples/sample1/dist2_mean.npy')
    meanDist_D_exp2 = DiagonalPreprocessing.genomic_distance_statistics(D_exp2, mode='freq')
    print(meanDist_D_exp2)
    m2 = len(meanDist_D_exp2)

    for log in [False]:
        if log:
            odir = osp.join(odir, 'log')
            if not osp.exists(odir):
                os.mkdir(odir, mode=0o755)
        for ar in [1.0, 1.5, 2.0]:
            for v in [6, 7, 8, 9, 10, 12]:
                for b in [160, 180, 200, 220, 240]:
                    grid_dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
                    if ar != 1.0:
                        grid_dir += f'_spheroid_{ar}'
                    max_ent_dir = grid_dir + '-max_ent10'
                    if osp.exists(max_ent_dir):
                        plot_meanDist_D(max_ent_dir,
                                        f'b_{b}_v_{v}_ar_{ar}')

                nan_rows = np.isnan(meanDist_D_exp)
                plt.plot(log_labels[~nan_rows][1:], meanDist_D_exp[~nan_rows][1:],
                            label='Experiment', color='k')

                nan_rows = np.isnan(meanDist_D_exp2)
                plt.plot(np.linspace(0, 30000*(m2-1), m2)[~nan_rows][1:], meanDist_D_exp2[~nan_rows][1:],
                            label='Experiment 2', color='k', ls=':')

                plt.ylabel('Distance (nm)', fontsize=16)
                plt.xlabel('Genomic Separation (bp)', fontsize=16)
                plt.legend()
                if log:
                    plt.xscale('log')
                    plt.savefig(osp.join(odir, f'd_s_max_ent_v_{v}_ar_{ar}_log.png'))
                else:
                    plt.savefig(osp.join(odir, f'd_s_max_ent_v_{v}_ar_{ar}.png'))
                plt.close()


def compare_meanDist_S():
    dataset = 'dataset_09_28_23'
    samples = [752, 2452]
    GNN_ID=496
    data_dir = osp.join('/home/erschultz/', dataset, 'samples')
    grid_root = 'optimize_grid_b_180_phi_0.008_spheroid_1.5'
    for s in samples:
        s_dir = osp.join(data_dir, f'sample{s}')
        Sgnn = np.load(osp.join(s_dir, f'{grid_root}-GNN{GNN_ID}/S.npy'))
        Sgt = np.load(osp.join(s_dir, 'S.npy'))
        for S, label in zip([Sgt, Sgnn],['Ground Truth', 'GNN']):
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
            print(label, meanDist[:5])
            plt.plot(meanDist, label = label)
        plt.legend()
        plt.xscale('log')
        plt.ylabel('Mean', fontsize=16)
        plt.xlabel('Off-diagonal Index', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(s_dir, 'meanDist_S.png'))
        plt.close()

def compare_meanDist_S2():
    '''Compare based on results/GNN_ID'''
    samples = [981, 3464]
    GNN_ID=556
    dataset='dataset_09_28_23'
    dir = f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{GNN_ID}'
    assert osp.exists(dir), dir
    for s in samples:
        print(s)
        s_dir = osp.join(dir, f'{dataset}_sample{s}/sample{s}')
        assert osp.exists(s_dir), s_dir
        Sgnn = np.loadtxt(osp.join(s_dir, 'energy_hat.txt'))
        Sgt = np.loadtxt(osp.join(s_dir, 'energy.txt'))
        for S, label in zip([Sgt, Sgnn],['Ground Truth', 'GNN']):
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
            print(label, meanDist[:5])

            plt.plot(meanDist, label = label)
        plt.legend()
        plt.xscale('log')
        plt.ylabel('Mean', fontsize=16)
        plt.xlabel('Off-diagonal Index', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(s_dir, 'meanDist_S.png'))
        plt.close()

        Sgnn = np.multiply(np.sign(Sgnn), np.exp(np.abs(Sgnn)) - 1)
        Sgt = np.multiply(np.sign(Sgt), np.exp(np.abs(Sgt)) - 1)
        for S, label in zip([Sgt, Sgnn],['Ground Truth', 'GNN']):
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
            print(label, meanDist[:5])

            plt.plot(meanDist, label = label)
        plt.legend()
        plt.xscale('log')
        plt.ylabel('Mean', fontsize=16)
        plt.xlabel('Off-diagonal Index', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(s_dir, 'meanDist_S_reg.png'))
        plt.close()

def check_GNN_S():
    GNN_ID=484
    dataset='dataset_04_28_23'; b=140; phi=0.03
    GNN_dir = f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{GNN_ID}'
    data_dir = f'/home/erschultz/{dataset}/samples'
    for sample in [324, 981, 1936, 2834, 3464]:
        print(sample)
        e_gt1 = np.loadtxt(osp.join(GNN_dir, f'{dataset}_sample{sample}-regular/sample{sample}-regular/energy.txt'))
        # e_gt1 = np.multiply(np.sign(e_gt1), np.exp(np.abs(e_gt1)) - 1)
        e_hat1 = np.loadtxt(osp.join(GNN_dir, f'{dataset}_sample{sample}-regular/sample{sample}-regular/energy_hat.txt'))
        # e_hat1 = np.multiply(np.sign(e_hat1), np.exp(np.abs(e_hat1)) - 1)
        # print(e_gt1[0, :10], e_hat1.shape)
        # print(e_hat1[0, :10], e_hat1.shape)
        diff = e_hat1 - e_gt1
        # print('MAE (e_gt, e_hat1)', np.mean(np.abs(diff)))


        e_gt2 = load_S(osp.join(data_dir, f'sample{sample}'))
        assert np.allclose(e_gt2, e_gt1, atol=1e-3, rtol=1e-3)
        e_hat2 = np.loadtxt(osp.join(data_dir, f'sample{sample}/optimize_grid_b_{b}_phi_{phi}-GNN{GNN_ID}/smatrix.txt'))
        # print(e_gt2[0, :10], e_hat1.shape)
        # print(e_hat2[0, :10])
        diff = e_hat2 - e_gt2
        # print('MAE (e_gt, e_hat2)', np.mean(np.abs(diff)))


        diff = e_hat2 - e_hat1
        print('MAE (e_hat2, e_hat1)', np.mean(np.abs(diff)))
        print()

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

def compare_cell_lines():
    odir='/home/erschultz/dataset_HCT116_RAD21_KO'
    dir1='/home/erschultz/dataset_HCT116_RAD21_KO/samples/sample1'
    dir2='/home/erschultz/dataset_HCT116_RAD21_KO/samples/sample88'
    cell_lines = ['HCT116_KO', 'HMEC']
    dirs = [dir1, dir2]
    for dir, cell_line in zip(dirs, cell_lines):
        y = np.load(osp.join(dir, 'y.npy'))
        y /= np.mean(y.diagonal())
        plot_matrix(y, osp.join(dir, f'y.png'), vmax='mean')

        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = cell_line)

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Beads', fontsize=16)
    plt.legend()
    plt.savefig(osp.join(odir, f'p_s_cell_line.png'))
    plt.close()

def soren_seq():
    seq = np.load('/home/erschultz/Downloads/HCT116_auxin_chr2_seqs20k.npy')
    seq = hic_utils.pool_seqs(seq, 10)
    seq = seq.T
    m, k = seq.shape
    print(seq.shape)

    fig, axes = plt.subplots(2, 3)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    for i, ax in enumerate(axes.flatten()):
        if i >= k:
            continue
        ax.plot(np.arange(0, m), seq[:, i])

    fig.supxlabel('Distance', fontsize=16)
    fig.supylabel('Label Value', fontsize=16)
    plt.tight_layout()
    plt.savefig('/home/erschultz/Downloads/soren_seq.png')
    plt.close()

if __name__ == '__main__':
    # compare_diag_fits()
    # compare_p_s_bonded3()
    compare_d_s_bonded()
    # compare_d_s_bonded2()
    # compare_d_s_max_ent()
    # compare_p_s_exp()
    # compare_meanDist_S2()
    # compare_cell_lines()
    # soren_seq()
    # visualize_max_ent_methods()
    # compare_xyz()
    # check_GNN_S()

    # split_dataset('dataset_09_28_23', 1, 0.36)
    # split_dataset('dataset_09_28_23', 10, 0.08)
    # split_dataset('dataset_09_28_23', 100, 0.01)
    # split_dataset2('dataset_09_28_23', 0.02)
