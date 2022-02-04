'''Intended for use with maxent.'''
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.utils import diagonal_preprocessing, load_final_max_ent_S
from data_summary_plots import genomic_distance_statistics
from plotting_functions import plotContactMap

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--k', type=int, help='number of bead labels')
    parser.add_argument('--save_npy', action='store_true', help='true to save y as .npy')
    parser.add_argument('--final_it', type=int, help='location of contact map')
    parser.add_argument('--replicate_folder', help='path to max_ent replicate folder')

    args.final_folder = osp.join(args.replicate_folder, f"iteration{args.final_it})
    args = parser.parse_args()
    return args

def main():
    args = getArgs()
    y_path = osp.join(args.final_folder, "production_out", "contacts.txt")

    if osp.exists(y_path)):
        y = np.loadtxt(y_path)[:args.m, :args.m]

    plotContactMap(y, ofile = osp.join(args.replicate_folder, 'y.png'), vmax = 'mean')

    load_final_max_ent_S(args.k, args.replicate_folder, args.final_folder)

    plotContactMap(s, ofile = osp.join(args.odir, 's.png'), title = 'S', vmax = 'max', vmin = 'min', cmap = 'blue-red')



    e_matrix_files = ['e.npy', 'e_matrix.txt']
    for e_matrix_file, load_fn in zip(e_matrix_files, load_fns):
        if osp.exists(e_matrix_file):
            e = load_fn(e_matrix_file)
            plotContactMap(e, ofile = osp.join(args.odir, 'e.png'), title = 'E', vmax = 'max', vmin = 'min', cmap = 'blue-red')
            break # don't plot twice



    if args.save_npy:
        np.save(osp.join(args.replicate_folder, 'y.npy'), y.astype(np.int16))
        np.save(osp.join(args.replicate_folder, 's.npy'), s)

        meanDist = genomic_distance_statistics(y)
        y_diag = diagonal_preprocessing(y, meanDist)
        plotContactMap(y_diag, ofile = osp.join(args.replicate_folder, 'y_diag.png'), vmax = 'max')
        np.save(osp.join(args.replicate_folder, 'y_diag.npy'), y_diag)

if __name__ == '__main__':
    main()
