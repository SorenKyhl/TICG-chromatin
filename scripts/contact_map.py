import argparse
import json
import math
import os
import os.path as osp
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import ArgparserConverter
from sequences_to_contact_maps.scripts.energy_utils import (
    calculate_diag_chi_step, s_to_E)
from sequences_to_contact_maps.scripts.load_utils import (load_E_S,
                                                          load_final_max_ent_S)
from sequences_to_contact_maps.scripts.plotting_utils import (
    plot_diag_chi, plot_matrix, plot_mean_dist, plot_mean_vs_genomic_distance)
from sequences_to_contact_maps.scripts.utils import DiagonalPreprocessing, crop


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
    parser.add_argument('--plot', action='store_true',
                        help='True to plot data, False to just save')

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
        args.final_it = final_it
        args.save_folder = args.replicate_folder

        path_split = args.replicate_folder.split(osp.sep)
        args.sample_folder = osp.join('/', *path_split[:path_split.index('samples')+2])
    return args

def main():
    args = getArgs()
    print(args)

    if args.random_mode:
        y_path = osp.join(args.sample_folder, 'data_out')
    else:
        y_path = osp.join(args.final_folder, "production_out")

    y_file = osp.join(y_path, 'contacts.txt')
    if osp.exists(y_file):
        y = crop(np.loadtxt(y_file), args.m)
        args.m = len(y)
    else:
        max_sweep = -1
        # look for contacts{sweep}.txt
        for file in os.listdir(y_path):
            if file.startswith('contacts') and file.endswith('.txt'):
                sweep = int(file[8:-4])
                if sweep > max_sweep:
                    max_sweep = sweep
        if max_sweep > 0:
            y = crop(np.loadtxt(osp.join(y_path, f'contacts{max_sweep}.txt')), args.m)
        else:
            raise Exception(f"y path does not exist: {y_path}")

    if args.plot:
        plot_matrix(y, ofile = osp.join(args.save_folder, 'y.png'), vmax = 'mean')
        p = y / np.mean(np.diagonal(y))
        plot_matrix(p, ofile = osp.join(args.save_folder, 'p.png'), vmax = 'mean')

    if args.random_mode:
        e, s = load_E_S(args.sample_folder, throw_exception = False)
    else:
        s = load_final_max_ent_S(args.replicate_folder, args.final_folder)
        e = s_to_E(s)

    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    y_diag = DiagonalPreprocessing.process(y, meanDist)

    if args.m < 5000 and args.plot:
        # takes a long time for large m and not really necessary
        plot_matrix(y_diag, ofile = osp.join(args.save_folder, 'y_diag.png'),
                    vmin = 'center1', cmap='blue-red')

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
    all_diag_chis = None
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
            all_diag_chis = np.loadtxt(file)
            diag_chi = all_diag_chis[-1]
            diag_chi_ref = None

    if diag_chi is not None:
        plot_diag_chi(config, args.save_folder,
                        ref = diag_chi_ref, ref_label = 'continuous')
        plot_diag_chi(config, args.save_folder,
                        ref = diag_chi_ref, ref_label = 'continuous',
                        logx = True)

    if all_diag_chis is not None:
        # plot gif of diag chis
        files = []
        ylim = (np.min(all_diag_chis), np.max(all_diag_chis))
        for i in range(1, len(all_diag_chis)):
            diag_chi_i = all_diag_chis[i]
            file = f'{i}.png'
            diag_chi_step = calculate_diag_chi_step(config, diag_chi_i)
            plot_diag_chi(None, args.save_folder,
                            logx = True, ofile = file,
                            diag_chis_step = diag_chi_step, ylim = ylim,
                            title = f'Iteration {i}')
            files.append(osp.join(args.save_folder, file))

        frames = []
        for filename in files:
            frames.append(imageio.imread(filename))

        imageio.mimsave(osp.join(args.save_folder, 'pchis_diag_step.gif'), frames, format='GIF', fps=2)

        # remove files
        for filename in files:
            os.remove(filename)


    # meanDist
    if args.plot:
        if 'diag_chis' in config.keys():
            diag_chi_step = calculate_diag_chi_step(config)
        else:
            diag_chi_step = None
        if args.random_mode:
            plot_mean_vs_genomic_distance(y, args.save_folder, 'meanDist.png',
                                            diag_chi_step)
            plot_mean_vs_genomic_distance(y, args.save_folder, 'meanDist.png',
                                            diag_chi_step, logx = True)

        else:
            meanDist_max_ent = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            print('meanDist_max_ent', meanDist_max_ent)
            y_gt_file = osp.join(args.replicate_folder, 'resources', 'y_gt.npy')
            if osp.exists(y_gt_file):
                y_gt = np.load(y_gt_file)
                meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob')
                print('meanDist_gt', meanDist_gt)
                mse = mean_squared_error(meanDist_max_ent, meanDist_gt)
                title = f'MSE: {np.round(mse, 9)}'
            else:
                meanDist_gt = None
                title = None

            if args.final_it == 1:
                sim_label = 'GNN'
                color = 'green'
            else:
                sim_label = 'Max Ent'
                color = 'blue'
            plot_mean_dist(meanDist_max_ent, args.save_folder, 'meanDist.png',
                            diag_chi_step, False, meanDist_gt, 'Reference', sim_label,
                            color, title)
            plot_mean_dist(meanDist_max_ent, args.save_folder, 'meanDist_log.png',
                            diag_chi_step, True, meanDist_gt, 'Reference', sim_label,
                            color, title)


if __name__ == '__main__':
    main()
