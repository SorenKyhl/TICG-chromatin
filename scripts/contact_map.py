import argparse
import json
import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seq2contact import (ArgparserConverter, DiagonalPreprocessing, crop,
                         get_diag_chi_step, load_E_S, load_final_max_ent_S,
                         plot_diag_chi, plot_matrix, plot_mean_dist,
                         plot_mean_vs_genomic_distance, s_to_E)


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

    # get config
    if args.random_mode:
        with open(osp.join(args.sample_folder, 'config.json'), 'r') as f:
            config = json.load(f)
    else:
        # find last iteration
        max_it = 1
        for f in os.listdir(args.replicate_folder):
            if f.startswith('iteration'):
                it = int(f[9:])
                if it > max_it:
                    max_it = it
        print(f'max_it = {max_it}')
        with open(osp.join(args.replicate_folder, f'iteration{max_it}', 'config.json'), 'r') as f:
            config = json.load(f)

    # diag chi
    diag_chi = None
    if args.random_mode:
        file = osp.join(args.sample_folder, 'diag_chis.npy')
        if osp.exists(file):
            diag_chi = np.load(file)
            file = osp.join(args.sample_folder, 'diag_chis_continuous.npy')
            if osp.exists(file):
                diag_chi_ref = np.load(file)
            else:
                diag_chi_ref = None
    else:
        file = osp.join(args.replicate_folder, 'chis_diag.txt')
        if osp.exists(file):
            diag_chi = np.loadtxt(file)[-1]
            diag_chi_ref = None

    if diag_chi is not None:
        if config['dense_diagonal_on']:
            plot_diag_chi(config, args.save_folder,
                            diag_chi_ref, 'continuous')
        else:
            plot_diag_chi(config, args.save_folder,
                            ref = diag_chi_ref, ref_label = 'continuous')

    # meanDist
    diag_chi_step = get_diag_chi_step(config)
    if args.random_mode:
        plot_mean_vs_genomic_distance(y, args.save_folder, 'meanDist.png',
                                        diag_chi_step)
        plot_mean_vs_genomic_distance(y, args.save_folder, 'meanDist.png',
                                        diag_chi_step, logx = True)

    else:
        meanDist_max_ent = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        y_gt = np.load(osp.join(args.sample_folder, 'y.npy'))
        meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob')

        plot_mean_dist(meanDist_max_ent, args.save_folder, 'meanDist.png',
                        diag_chi_step, False, meanDist_gt)
        plot_mean_dist(meanDist_max_ent, args.save_folder, 'meanDist_norm.png',
                        diag_chi_step, False, meanDist_gt, True)
        plot_mean_dist(meanDist_max_ent, args.save_folder, 'meanDist_log.png',
                        diag_chi_step, True, meanDist_gt)
        plot_mean_dist(meanDist_max_ent, args.save_folder, 'meanDist_log_norm.png',
                        diag_chi_step, True, meanDist_gt, True)

if __name__ == '__main__':
    main()
