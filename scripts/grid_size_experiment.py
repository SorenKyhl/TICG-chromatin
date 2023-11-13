import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import plot_matrix, plot_mean_dist

sys.path.append('/home/erschultz/TICG-chromatin')
import max_ent
from scripts.data_generation.modify_maxent import get_samples

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
    for p in [0.2, 0.25, 0.3, 0.35]:
        for i in samples:
            mapping.append((dataset, i, f'samples_p{p}',
                            180, None, 8, None, 1.5))

    print(len(mapping))
    print(mapping)

    with mp.Pool(4) as p:
        p.starmap(max_ent.fit, mapping)

def analysis():
    probabilities = [0.2, 0.25, 0.3, 0.35]
    dir = '/home/erschultz/grid_size_analysis'
    L_list = []
    chi_list = []
    for p in probabilities:
        p_dir = f'{dir}/samples_p{p}'
        s_dir = osp.join(p_dir, 'sample201')
        final = osp.join(s_dir, 'optimize_grid_b_180_v_8_spheroid_1.5-max_ent10/iteration30')
        S = np.load(osp.join(final, 'S.npy'))
        L = np.load(osp.join(final, 'L.npy'))
        with open(osp.join(final, 'production_out/config.json')) as f:
            config = json.load(f)
            chis = np.array(config['chis'])
            chi_list.append(chis)
        L_list.append(L)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, mode='freq')
        plt.plot(meanDist, label = p)

    plt.xscale('log')
    plt.legend(title='p(s=1)')
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('Effective Diagonal Parameter', fontsize=16)
    plt.savefig(osp.join(dir, 'meanDist_S.png'))
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
    plt.xlabel(f'p={probabilities[0]}', fontsize=16)
    plt.ylabel(f'p={probabilities[-1]}', fontsize=16)
    plt.title(r'$L_{ij}$')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'affine_L.png'))
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
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(f'p={probabilities[0]}', fontsize=16)
    plt.ylabel(f'p={probabilities[-1]}', fontsize=16)
    plt.title(r'$\chi_{ij}$')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'affine_chi.png'))
    plt.close()





def main():
    # make_samples()
    # fit_max_ent()
    analysis()

if __name__ == '__main__':
    main()
