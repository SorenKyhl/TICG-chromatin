import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seq2contact import (ArgparserConverter, DiagonalPreprocessing, crop,
                         load_E_S, load_final_max_ent_S, plot_matrix, s_to_E)


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    parser.add_argument('--m', type=int, default=1024,
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

if __name__ == '__main__':
    main()
