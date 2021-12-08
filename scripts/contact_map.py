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

from neural_net_utils.utils import diagonal_preprocessing
from data_summary_plots import genomic_distance_statistics
from plotting_functions import plotContactMap

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--save_npy', action='store_true', help='true to save y as .npy')
    parser.add_argument('--ifile', default=osp.join('data_out','contacts.txt'), help='location of contact map')
    parser.add_argument('--odir', default='', help='path to output directory (default is wd)')

    args = parser.parse_args()
    return args

def main():
    args = getArgs()
    if args.ifile.endswith('.txt'):
        y = np.loadtxt(args.ifile)[:args.m, :args.m]
    elif args.ifile.endswith('.npy'):
        y = np.load(args.ifile)[:args.m, :args.m]
    else:
        raise Exception("Unrecognized extension: {}".format(args.ifile))


    plotContactMap(y, ofile = osp.join(args.odir, 'y.png'), vmax = 'mean')

    load_fns = [np.load, np.loadtxt]
    s_matrix_files = ['s.npy', 's_matrix.txt']
    for s_matrix_file, load_fn in zip(s_matrix_files, load_fns):
        if osp.exists(s_matrix_file):
            s = load_fn(s_matrix_file)
            plotContactMap(s, ofile = osp.join(args.odir, 's.png'), title = 'S', vmax = 'max', vmin = 'min', cmap = 'blue-red')
            break # don't plot twice

    e_matrix_files = ['e.npy', 'e_matrix.txt']
    for e_matrix_file, load_fn in zip(e_matrix_files, load_fns):
        if osp.exists(e_matrix_file):
            e = load_fn(e_matrix_file)
            plotContactMap(e, ofile = osp.join(args.odir, 'e.png'), title = 'E', vmax = 'max', vmin = 'min', cmap = 'blue-red')
            break # don't plot twice

    if args.save_npy:
        np.save(osp.join(args.odir, 'y.npy'), y.astype(np.int16))

        meanDist = genomic_distance_statistics(y)
        y_diag_instance = diagonal_preprocessing(y, meanDist)
        plotContactMap(y_diag_instance, ofile = osp.join(args.odir, 'y_diag.png'), vmax = 'max')
        np.save(osp.join(args.odir, 'y_diag.npy'), y_diag_instance)

if __name__ == '__main__':
    main()
