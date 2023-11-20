import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import matplotlib.pyplot as plt
import max_ent
import numpy as np
from data_generation.modify_maxent import get_samples
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_diag_chi_step
from pylib.utils.plotting_utils import plot_matrix, plot_mean_dist
from pylib.utils.utils import load_json
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder

EXP_DATASET='dataset_02_04_23'
OVERWRITE=False

def make_samples():
    probabilities = [0.2, 0.25, 0.3, 0.35]
    dir = '/home/erschultz/grid_size_analysis'
    if not osp.exists(dir):
        os.mkdir(dir, mode=0o755)
    for p in probabilities:
        p_dir = f'{dir}/samples_p{p}'
        if not osp.exists(p_dir):
            os.mkdir(p_dir, mode=0o755)

    samples, _ = get_samples(EXP_DATASET, train=True)
    for s_exp in samples[:1]:
        print(s_exp)
        exp_dir = f'/home/erschultz/{EXP_DATASET}/samples/sample{s_exp}'
        y = np.load(osp.join(exp_dir, 'y.npy'))
        y /= np.mean(y.diagonal())
        m = len(y)
        print(np.mean(y.diagonal()))

        for p in probabilities:
            p_dir = f'{dir}/samples_p{p}'
            print(f'probability={p}')
            odir = osp.join(p_dir, f'sample{s_exp}')
            if osp.exists(odir):
                if OVERWRITE:
                    shutil.rmtree(odir)
                else:
                    continue
            os.mkdir(odir, mode = 0o755)

            diag = y.diagonal().copy()
            y_copy = y.copy()
            np.fill_diagonal(y_copy, 0)
            y_copy /= np.mean(y_copy.diagonal(offset=1))
            y_copy *= p
            np.fill_diagonal(y_copy, diag)

            print(np.mean(y_copy.diagonal(offset=0)))
            print(np.mean(y_copy.diagonal(offset=1)))

            np.save(osp.join(odir, 'y.npy'), y_copy)
            plot_matrix(y_copy, osp.join(odir, 'y.png'), vmax = 'mean')

            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_copy, 'prob')
            plot_mean_dist(meanDist, odir, 'meanDist_log.png',
                            None, True)

def fit_max_ent():
    dataset='grid_size_analysis'
    samples, _ = get_samples(EXP_DATASET, train=True)
    N = 1
    samples = samples[:N]

    mapping = []
    b=180;ar=1.5;phi=None;v=8
    contacts_distance=False
    for p in [0.2, 0.25, 0.3, 0.35]:
        for i in samples:
            mapping.append((dataset, i, f'samples_p{p}',
                            b, phi, v, None, ar, 'gaussian', 10, contacts_distance))

    print(len(mapping))
    print(mapping)

    with mp.Pool(4) as p:
        p.starmap(max_ent.check, mapping)

def analysis():
    probabilities = [0.2, 0.25, 0.3, 0.35]
    dir = '/home/erschultz/grid_size_analysis'
    L_list = []
    L_gs_list = []
    chi_list = []
    chi_gs_list = []
    S_list = []
    gs_list = []
    for p in probabilities:
        p_dir = f'{dir}/samples_p{p}'
        s_dir = osp.join(p_dir, 'sample201')
        final = osp.join(s_dir, 'optimize_grid_b_180_v_8_spheroid_1.5-max_ent10/iteration30')
        S = np.load(osp.join(final, 'S.npy'))
        L = np.load(osp.join(final, 'L.npy'))
        with open(osp.join(final, 'production_out/config.json')) as f:
            config = json.load(f)
            chis = np.array(config['chis'])
            gs = config['grid_size']
            gs_list.append(np.round(gs, 1))
            print(gs)
            chi_list.append(chis)
            chi_gs_list.append(chis*gs**3)
        L_list.append(L)
        L_gs_list.append(L/gs**3)
        S_list.append(S)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, mode='freq')
        plt.plot(meanDist, label = p)

    plt.xscale('log')
    plt.legend(title='p(s=1)')
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('Effective Diagonal Parameter', fontsize=16)
    plt.savefig(osp.join(dir, 'meanDist_S.png'))
    plt.close()

    # test affine transformation S
    x = S_list[0].flatten()
    # x -= np.min(x)
    y = S_list[-1].flatten()
    # y -= np.min(y)
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    # plt.axline((0, b), slope=m, color = 'k', ls = '--')
    X = np.linspace(np.min(x), np.max(x))
    Y = m*X+b
    plt.plot(X, Y, color = 'k', ls = '--')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(f'p={probabilities[0]}, $\Delta$={gs_list[0]}', fontsize=16)
    plt.ylabel(f'p={probabilities[-1]}, $\Delta$={gs_list[1]}', fontsize=16)
    plt.title(r'$S_{ij}$')
    # plt.ylim(-100, 200)
    # plt.xlim(-50, 75)
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'affine_S.png'))
    plt.close()

    # test affine transformation L
    x = L_list[0].flatten()
    # x -= np.min(x)
    y = L_list[-1].flatten()
    # y -= np.min(y)
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    # plt.axline((0, b), slope=m, color = 'k', ls = '--')
    X = np.linspace(np.min(x), np.max(x))
    Y = m*X+b
    plt.plot(X, Y, color = 'k', ls = '--')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(f'p={probabilities[0]}, $\Delta$={gs_list[0]}', fontsize=16)
    plt.ylabel(f'p={probabilities[-1]}, $\Delta$={gs_list[1]}', fontsize=16)
    plt.title(r'$L_{ij}$')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'affine_L.png'))
    plt.close()

    # test affine transformation L_gs
    x = L_gs_list[0].flatten()
    # x -= np.min(x)
    y = L_gs_list[-1].flatten()
    # y -= np.min(y)
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    # plt.axline((0, b), slope=m, color = 'k', ls = '--')
    X = np.linspace(np.min(x), np.max(x))
    Y = m*X+b
    plt.plot(X, Y, color = 'k', ls = '--')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(f'p={probabilities[0]}, $\Delta$={gs_list[0]}', fontsize=16)
    plt.ylabel(f'p={probabilities[-1]}, $\Delta$={gs_list[1]}', fontsize=16)
    plt.title(r'$L_{ij}/\Delta^3$')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'affine_L_gs.png'))
    plt.close()


    # test affine transformation chi
    x = chi_list[0].flatten()
    # x -= np.min(x)
    y = chi_list[-1].flatten()
    # y -= np.min(y)
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    # plt.axline((0, b), slope=m, color = 'k', ls = '--')
    X = np.linspace(np.min(x), np.max(x))
    Y = m*X+b
    plt.plot(X, Y, color = 'k', ls = '--')
    plt.plot(X, X, color = 'k')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(f'p={probabilities[0]}, $\Delta$={gs_list[0]}', fontsize=16)
    plt.ylabel(f'p={probabilities[-1]}, $\Delta$={gs_list[1]}', fontsize=16)
    plt.title(r'$\chi_{ij}$')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'affine_chi.png'))
    plt.close()

    # test affine transformation chi_gs
    x = chi_gs_list[0].flatten()
    # x -= np.min(x)
    y = chi_gs_list[-1].flatten()
    # y -= np.min(y)
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    # plt.axline((0, b), slope=m, color = 'k', ls = '--')
    X = np.linspace(np.min(x), np.max(x))
    Y = m*X+b
    plt.plot(X, Y, color = 'k', ls = '--')
    plt.plot(X, X, color = 'k')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(f'p={probabilities[0]}, $\Delta$={gs_list[0]}', fontsize=16)
    plt.ylabel(f'p={probabilities[-1]}, $\Delta$={gs_list[1]}', fontsize=16)
    plt.title(r'$\chi_{ij}\times\Delta^3$')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'affine_chi_gs.png'))
    plt.close()


def grid_200_analysis():
    probabilities = [0.2, 0.25, 0.3, 0.35]
    dir = '/home/erschultz/grid_size_analysis'
    max_ent_root = 'optimize_grid_b_180_v_8_spheroid_1.5-max_ent10_grid200'
    for p in probabilities:
        p_dir = f'{dir}/samples_p{p}'
        final = get_final_max_ent_folder(osp.join(p_dir, 'sample201', max_ent_root))
        config = load_json(osp.join(final, 'config.json'))
        gs = config['grid_size']
        y = np.load(osp.join(final, 'y.npy'))
        S = np.load(osp.join(final, 'S.npy'))
        diag_chis = calculate_diag_chi_step(config)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
        plt.plot(meanDist, label=p)

    plt.xscale('log')
    plt.legend(title='P(s=1)')
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('Effective Diagonal Parameter', fontsize=16)
    plt.savefig(osp.join(dir, 'diag_chis_gs200.png'))
    plt.close()

def start_exp_analysis():
    starts = [5]
    dir = '/home/erschultz/dataset_06_29_23/samples/'
    samples = [1,2,3,4,5,6,7, 16]
    odir = '/home/erschultz/dataset_06_29_23'
    bonded_root = 'optimize_grid_b_180_v_8_spheroid_1.5'

    fig, axes = plt.subplots(4, 2)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    for s, ax in zip(samples, axes.flatten()):
        y_b = np.load(osp.join(dir, f'sample{s}', bonded_root, 'y.npy'))
        meanDist_b = DiagonalPreprocessing.genomic_distance_statistics(y_b, 'prob')
        ax.plot(meanDist_b, c='b')

        y_exp = np.load(osp.join(dir, f'sample{s}/y.npy'))
        meanDist_exp = DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob')
        ax.plot(meanDist_exp, c='k')
        rmse = mean_squared_error(meanDist_b, meanDist_exp, squared = False)
        ax.set_title(f'Sample {s}, MSE={np.round(rmse, 4)}')
        ax.axhline(1e-3, c='gray', ls=":")
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1e-1)
        ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(osp.join(odir, f'bonded_vs_exp.png'))
    plt.close()

    for start in starts:
        max_ent_root = f'{bonded_root}-max_ent10_start{start}'
        for s in samples:
            final = get_final_max_ent_folder(osp.join(dir, f'sample{s}', max_ent_root))
            config = load_json(osp.join(final, 'config.json'))
            gs = config['grid_size']
            y = np.load(osp.join(final, 'y.npy'))
            meanDist_y = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            p_1 = np.round(meanDist_y[1], 2)
            S = np.load(osp.join(final, 'S.npy'))
            meanDist_S = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
            plt.plot(meanDist_S, label=f'{s}, $\Delta$={int(gs)}, p(s=1)={p_1}')

        plt.xscale('log')
        plt.legend(title='Sample', fontsize=14)
        plt.xlabel('Distance', fontsize=16)
        plt.ylabel('Effective Diagonal Parameter', fontsize=16)
        plt.savefig(osp.join(odir, f'diag_chis_start{start}.png'))
        plt.close()

def start_analysis():
    starts = [0,3,4,5,6,7,10]
    probabilities = [0.2, 0.25, 0.3, 0.35]
    dir = '/home/erschultz/grid_size_analysis'
    odir = dir
    bonded_root = 'optimize_grid_b_180_v_8_spheroid_1.5'

    # fig, axes = plt.subplots(2, 2)
    # fig.set_figheight(12)
    # fig.set_figwidth(12)
    # for p, ax in zip(probabilities, axes.flatten()):
    #     y_b = np.load(osp.join(dir, f'samples_p{p}/sample201', bonded_root, 'y.npy'))
    #     meanDist_b = DiagonalPreprocessing.genomic_distance_statistics(y_b, 'prob')
    #     ax.plot(meanDist_b, c='b')
    #
    #     y_exp = np.load(osp.join(dir, f'samples_p{p}/sample201/y.npy'))
    #     meanDist_exp = DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob')
    #     ax.plot(meanDist_exp, c='k')
    #     rmse = mean_squared_error(meanDist_b, meanDist_exp, squared = False)
    #     ax.set_title(f'P {p}, MSE={np.round(rmse, 4)}')
    #     ax.axhline(1e-3, c='gray', ls=":")
    #     ax.set_yscale('log')
    #     ax.set_ylim(1e-4, 0.4)
    #     ax.set_xscale('log')
    # plt.tight_layout()
    # plt.savefig(osp.join(odir, f'bonded_vs_exp.png'))
    # plt.close()

    for start in starts:
        if start == 0:
            max_ent_root = f'{bonded_root}-max_ent10'
        else:
            max_ent_root = f'{bonded_root}-max_ent10_start{start}'
        for p in probabilities:
            final = get_final_max_ent_folder(osp.join(dir, f'samples_p{p}/sample201', max_ent_root))
            config = load_json(osp.join(final, 'config.json'))
            gs = config['grid_size']
            S_file = osp.join(final, 'S.npy')
            if not osp.exists(S_file):
                continue
            S = np.load(S_file)
            meanDist_S = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
            plt.plot(meanDist_S, label=f'{p}, $\Delta$={int(gs)}')

        plt.ylim(-5, 50)
        plt.xscale('log')
        plt.legend(title='Sample', fontsize=14)
        plt.xlabel('Distance', fontsize=16)
        plt.ylabel('Effective Diagonal Parameter', fontsize=16)
        plt.savefig(osp.join(odir, f'diag_chis_start{start}.png'))
        plt.close()

    for start in starts:
        if start == 0:
            max_ent_root = f'{bonded_root}-max_ent10'
        else:
            max_ent_root = f'{bonded_root}-max_ent10_start{start}'
        for p in probabilities:
            me_dir = osp.join(dir, f'samples_p{p}/sample201', max_ent_root)
            final = get_final_max_ent_folder(me_dir)
            config = load_json(osp.join(final, 'config.json'))
            gs = config['grid_size']
            chis = np.loadtxt(osp.join(me_dir, 'chis.txt'))
            plt.plot(chis, label=f'{p}, $\Delta$={int(gs)}')

        plt.ylim(-800, 400)
        plt.legend(title='Sample', fontsize=14)
        plt.xlabel('Index', fontsize=16)
        plt.ylabel('Plaid Chi', fontsize=16)
        plt.savefig(osp.join(odir, f'plaid_chis_start{start}.png'))
        plt.close()


if __name__ == '__main__':
    # make_samples()
    # fit_max_ent()
    # analysis()
    # grid_200_analysis()
    start_analysis()
