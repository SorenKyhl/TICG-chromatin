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
    calculate_D, calculate_diag_chi_step, calculate_S)
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_final_max_ent_L, load_L)
from sequences_to_contact_maps.scripts.plotting_utils import (
    plot_diag_chi, plot_matrix, plot_mean_dist, plot_mean_vs_genomic_distance)
from sequences_to_contact_maps.scripts.utils import DiagonalPreprocessing, crop


def getArgs(sample_folder=''):
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    parser.add_argument('--m', type=int, default=-1,
                        help='number of particles (-1 to infer)')
    parser.add_argument('--save_npy', action='store_true',
                        help='true to save y as .npy')
    parser.add_argument('--random_mode', action='store_true',
                        help='true for random_mode, default is max_ent mode')
    parser.add_argument('--plot', action='store_true',
                        help='True to plot data, False to just save')

    # random_mode
    parser.add_argument('--sample_folder', type=str, default=sample_folder,
                        help='path to sample folder')

    # max_ent mode
    parser.add_argument('--replicate_folder',
                        help='path to max_ent replicate folder')

    args = parser.parse_args()
    if args.random_mode:
        args.save_folder = args.sample_folder
    elif args.replicate_folder is not None:
        args.final_folder, args.final_it = get_final_max_ent_folder(args.replicate_folder, return_it = True)
        args.save_folder = args.replicate_folder

        path_split = args.replicate_folder.split(osp.sep)
        args.sample_folder = osp.join('/', *path_split[:path_split.index('samples')+2])
    return args

def plot_all(args):
    if args.random_mode:
        y_path = osp.join(args.sample_folder, 'data_out')
        if not osp.exists(y_path):
            y_path = args.sample_folder
    else:
        y_path = osp.join(args.final_folder, "production_out")
        if not osp.exists(y_path):
            y_path = args.final_folder


    # get y
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

    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    y_diag = DiagonalPreprocessing.process(y, meanDist)

    # get L
    if args.random_mode:
        L = load_L(args.sample_folder, throw_exception = False)
    else:
        L = load_final_max_ent_L(args.replicate_folder, args.final_folder)

    # get config
    if args.random_mode:
        with open(osp.join(args.sample_folder, 'config.json'), 'r') as f:
            config = json.load(f)
    else:
        with open(osp.join(args.final_folder, 'config.json'), 'r') as f:
            config = json.load(f)

    # diag chi
    diag_chi = None
    all_diag_chis = None
    diag_chi_step = None
    D = None
    if args.random_mode:
        file = osp.join(args.sample_folder, 'diag_chis.npy')
        if osp.exists(file):
            diag_chi = np.load(file)
            file = osp.join(args.sample_folder, 'diag_chis_continuous.npy')
            if osp.exists(file):
                diag_chi_ref = np.load(file)
            else:
                diag_chi_ref = None
    elif args.replicate_folder is not None:
        file = osp.join(args.replicate_folder, 'chis_diag.txt')
        if osp.exists(file):
            all_diag_chis = np.loadtxt(file)
            diag_chi = all_diag_chis[-1]
            diag_chi_ref = None
    if diag_chi is not None:
        diag_chi_step = calculate_diag_chi_step(config, diag_chi)
        D = calculate_D(diag_chi_step)

    # get S
    S = None
    if L is not None:
        S = calculate_S(L, D)
    elif args.replicate_folder is not None:
        file = osp.join(args.replicate_folder, 'resources/S.npy')
        if osp.exists(file):
            S = np.load(file)

    if args.plot:
        plot_matrix(y, ofile = osp.join(args.save_folder, 'y.png'), vmax = 'mean')
        p = y / np.mean(np.diagonal(y))
        plot_matrix(p, ofile = osp.join(args.save_folder, 'p.png'), vmax = 'mean')

        if diag_chi is not None:
            plot_diag_chi(config, args.save_folder,
                            ref = diag_chi_ref, ref_label = 'continuous')
            plot_diag_chi(config, args.save_folder,
                            ref = diag_chi_ref, ref_label = 'continuous',
                            logx = True)

        # plot gif of diag chis
        if all_diag_chis is not None:
            files = []
            ylim = (np.min(all_diag_chis), np.max(all_diag_chis))
            for i in range(1, len(all_diag_chis)):
                diag_chi_i = all_diag_chis[i]
                file = f'{i}.png'
                diag_chi_step_i = calculate_diag_chi_step(config, diag_chi_i)
                plot_diag_chi(None, args.save_folder,
                                logx = True, ofile = file,
                                diag_chis_step = diag_chi_step_i, ylim = ylim,
                                title = f'Iteration {i}')
                files.append(osp.join(args.save_folder, file))

            frames = []
            for filename in files:
                frames.append(imageio.imread(filename))

            imageio.mimsave(osp.join(args.save_folder, 'pchis_diag_step.gif'), frames, format='GIF', fps=2)

            # remove files
            for filename in files:
                os.remove(filename)

        # plot energy matrices
        if args.m < 5000:
            # takes a long time for large m and not really necessary
            plot_matrix(y_diag, ofile = osp.join(args.save_folder, 'y_diag.png'),
                        vmin = 'center1', cmap='blue-red')

            if S is not None:
                plot_matrix(S, ofile = osp.join(args.save_folder, 'S.png'), title = 'S',
                            vmax = 'max', vmin = 'min', cmap = 'blue-red')

            if L is not None:
                plot_matrix(L, ofile = osp.join(args.save_folder, 'L.png'), title = 'L',
                            vmax = 'max', vmin = 'min', cmap = 'blue-red')

        # meanDist
        if args.random_mode:
            y_gt_file = osp.join(args.sample_folder, 'resources', 'y_gt.npy')
            if osp.exists(y_gt_file):
                y_gt = np.load(y_gt_file)
                meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob')
                meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
                print('meanDist_gt', meanDist_gt.shape)
                rmse = mean_squared_error(meanDist, meanDist_gt, squared = False)
                title = f'RMSE: {np.round(rmse, 9)}'
                plot_mean_dist(meanDist, args.save_folder, 'meanDist_log_ref.png',
                                diag_chi_step, True, meanDist_gt, 'Reference', 'Sim',
                                'blue', title)
                plot_mean_dist(meanDist, args.save_folder, 'meanDist_ref.png',
                                diag_chi_step, False, meanDist_gt, 'Reference', 'Sim',
                                'blue', title)

            plot_mean_vs_genomic_distance(y, args.save_folder, 'meanDist.png',
                                            diag_chi_step)
            plot_mean_vs_genomic_distance(y, args.save_folder, 'meanDist_log.png',
                                            diag_chi_step, logx = True)

        else:
            meanDist_max_ent = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            print('meanDist_max_ent', meanDist_max_ent[:10])
            if args.replicate_folder is not None:
                y_gt_file = osp.join(args.replicate_folder, 'resources', 'y_gt.npy')
            else:
                y_gt_file = osp.join(osp.split(args.save_folder)[0],  'y.npy')
            if osp.exists(y_gt_file):
                y_gt = np.load(y_gt_file)
                meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob')
                print('meanDist_gt', meanDist_gt[:10])
                rmse = mean_squared_error(meanDist_max_ent, meanDist_gt, squared = False)
                title = f'RMSE: {np.round(rmse, 9)}'
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

    if args.save_npy:
        np.save(osp.join(args.save_folder, 'y.npy'), y.astype(np.int16))
        np.save(osp.join(args.save_folder, 'y_diag.npy'), y_diag)
        if S is not None:
            np.save(osp.join(args.save_folder, 'S.npy'), S)
        if L is not None:
            np.save(osp.join(args.save_folder, 'L.npy'), L)

def main():
    args = getArgs()
    print(args)
    plot_all(args)

def plot_random(folder):
    args = getArgs()
    args.random_mode = True
    args.save_npy = True
    args.plot = True
    args.sample_folder = folder
    args.save_folder = folder
    print(args)
    plot_all(args)

def plot_max_ent(folder):
    args = getArgs()
    args.save_npy = True
    args.plot = True
    args.final_folder, args.final_it = get_final_max_ent_folder(folder, return_it = True)
    args.save_folder = folder
    print(args)
    plot_all(args)

if __name__ == '__main__':
    main()
