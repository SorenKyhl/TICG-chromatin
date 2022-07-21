import argparse
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seq2contact import (ArgparserConverter, DiagonalPreprocessing, crop,
                         load_E_S, load_final_max_ent_S, plot_diag_chi,
                         plot_matrix, plot_mean_vs_genomic_distance, s_to_E)


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    parser.add_argument('--m', type=int, default=-1,
                        help='number of particles (-1 to infer)')
    parser.add_argument('--k', type=AC.str2int,
                        help='number of bead labels')
    parser.add_argument('--save_npy', action='store_true',
                        help='true to save y as .npy')
    parser.add_argument('--random_mode', action='store_true',
                        help='true for random_mode, default is max_ent mode')

    # random_mode
    parser.add_argument('--sample_folder', type=str, default='',
                        help='path to sample folder')

    # max_ent mode
    parser.add_argument('--replicate_folder',
                        help='path to max_ent replicate folder')

    args = parser.parse_args()
    if args.random_mode:
        args.save_folder = args.sample_folder
    else:
        final_it = -1
        for file in os.listdir(args.replicate_folder):
            if file.startswith('iteration'):
                it = int(file[9:])
                if it > final_it:
                    final_it = it
        args.final_folder = osp.join(args.replicate_folder, f"iteration{final_it}")
        args.save_folder = args.replicate_folder

        path_split = args.replicate_folder.split(osp.sep)
        args.sample_folder = osp.join('/', *path_split[:path_split.index('samples')+2])
    return args

def main():
    args = getArgs()
    print(args)

    if args.random_mode:
        y_path = osp.join(args.sample_folder, 'data_out', 'contacts.txt')
    else:
        y_path = osp.join(args.final_folder, "production_out", "contacts.txt")

    if osp.exists(y_path):
        y = crop(np.loadtxt(y_path), args.m)
        args.m = len(y)
    else:
        raise Exception(f"y path does not exist: {y_path}")

    plot_matrix(y, ofile = osp.join(args.save_folder, 'y.png'), vmax = 'mean')

    if args.random_mode:
        if args.k == 0:
            throw = False
        else:
            throw = True
        e, s = load_E_S(args.sample_folder, throw_exception = throw)
    else:
        s = load_final_max_ent_S(args.replicate_folder, args.final_folder)
        e = s_to_E(s)

    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    y_diag = DiagonalPreprocessing.process(y, meanDist)

    if args.m < 5000:
        # takes a long time for large m and not really necessary
        plot_matrix(y_diag, ofile = osp.join(args.save_folder, 'y_diag.png'),
                    vmax = 'max')

        if s is not None:
            plot_matrix(s, ofile = osp.join(args.save_folder, 's.png'), title = 'S',
                        vmax = 'max', vmin = 'min', cmap = 'blue-red')

        if e is not None:
            plot_matrix(e, ofile = osp.join(args.save_folder, 'e.png'), title = 'E',
                        vmax = 'max', vmin = 'min', cmap = 'blue-red')

    if args.save_npy:
        np.save(osp.join(args.save_folder, 'y.npy'), y.astype(np.int16))
        np.save(osp.join(args.save_folder, 'y_diag.npy'), y_diag)
        if s is not None:
            np.save(osp.join(args.save_folder, 's.npy'), s)

    # diag chi
    if args.random_mode:
        file = osp.join(args.sample_folder, 'diag_chis.npy')
        if osp.exists(file):
            diag_chi = np.load(file)
            with open(osp.join(args.sample_folder, 'config.json'), 'r') as f:
                config = json.load(f)
            file = osp.join(args.sample_folder, 'diag_chis_continuous.npy')
            if osp.exists(file):
                diag_chis_continuous = np.load(file)
            else:
                diag_chis_continuous = None
            plot_diag_chi(diag_chi, args.m, args.save_folder, config['dense_diagonal_on'],
                            config['dense_diagonal_cutoff'],
                            config['dense_diagonal_loading'],
                            diag_chis_continuous)
    else:
        file = osp.join(args.replicate_folder, 'chis_diag.txt')
        if osp.exists(file):
            diag_chi = np.loadtxt(file)
            diag_chi_gt = np.load(osp.join(args.sample_folder, 'diag_chis_continuous.npy'))
            with open(osp.join(args.replicate_folder, 'resources/config.json'), 'r') as f:
                config = json.load(f)
            plot_diag_chi(diag_chi[-1], args.m, args.save_folder, config['dense_diagonal_on'],
                            config['dense_diagonal_cutoff'],
                            config['dense_diagonal_loading'], diag_chi_gt)
        else:
            diag_chi = None

    # meanDist
    if args.random_mode:
        plot_mean_vs_genomic_distance(y, args.save_folder, diag_chi, 'meanDist.png',
                                    config['dense_diagonal_on'],
                                    config['dense_diagonal_cutoff'],
                                    config['dense_diagonal_loading'])
    else:
        meanDist_max_ent = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        print(meanDist_max_ent)
        y_gt = np.load(osp.join(args.sample_folder, 'y.npy'))
        meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob')
        print(meanDist_gt)

        fig, ax = plt.subplots()
        ax.plot(meanDist_max_ent, label = 'max ent')
        ax.plot(meanDist_gt, label = 'gt')
        ax.set_yscale('log')
        # ax.set_xscale('log')

        ax.set_ylabel('Contact Probability', fontsize = 16)
        ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(args.save_folder, 'meanDist.png'))
        plt.close()

if __name__ == '__main__':
    main()
