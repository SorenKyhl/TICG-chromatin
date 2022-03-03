import argparse
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seq2contact import (crop, diagonal_preprocessing,
                         genomic_distance_statistics, load_E_S,
                         load_final_max_ent_S, plotContactMap, str2int)


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--m', type=int, default=1024, help='number of particles (-1 to infer)')
    parser.add_argument('--k', type=str2int, help='number of bead labels')
    parser.add_argument('--save_npy', action='store_true', help='true to save y as .npy')
    parser.add_argument('--random_mode', action='store_true', help='true for random_mode, default is max_ent mode')

    # random_mode
    parser.add_argument('--sample_folder', type=str, default='', help='path to sample folder')

    # max_ent mode
    parser.add_argument('--final_it', type=int, help='location of contact map')
    parser.add_argument('--replicate_folder', help='path to max_ent replicate folder')

    args = parser.parse_args()
    if args.random_mode:
        args.save_folder = args.sample_folder
    else:
        args.final_folder = osp.join(args.replicate_folder, f"iteration{args.final_it}")
        args.save_folder = args.replicate_folder
    return args

def main():
    args = getArgs()

    if args.random_mode:
        y_path = osp.join(args.sample_folder, 'data_out', 'contacts.txt')
    else:
        y_path = osp.join(args.final_folder, "production_out", "contacts.txt")

    if osp.exists(y_path):
        y = crop(np.loadtxt(y_path), args.m)
    else:
        raise Exception(f"y path does not exist: {y_path}")

    plotContactMap(y, ofile = osp.join(args.save_folder, 'y.png'), vmax = 'mean')

    if args.random_mode:
        e, s = load_E_S(args.sample_folder)
    else:
        s = load_final_max_ent_S(args.k, args.replicate_folder, args.final_folder)
        e = None

    if s is not None:
        plotContactMap(s, ofile = osp.join(args.save_folder, 's.png'), title = 'S', vmax = 'max', vmin = 'min', cmap = 'blue-red')

    if e is not None:
        # TODO this should work every time
        plotContactMap(e, ofile = osp.join(args.save_folder, 'e.png'), title = 'E', vmax = 'max', vmin = 'min', cmap = 'blue-red')


    meanDist = genomic_distance_statistics(y)
    y_diag = diagonal_preprocessing(y, meanDist)
    plotContactMap(y_diag, ofile = osp.join(args.save_folder, 'y_diag.png'), vmax = 'max')


    if args.save_npy:
        np.save(osp.join(args.save_folder, 'y.npy'), y.astype(np.int16))
        np.save(osp.join(args.save_folder, 's.npy'), s)
        np.save(osp.join(args.save_folder, 'y_diag.npy'), y_diag)

if __name__ == '__main__':
    main()
