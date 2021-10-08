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
        '/home/eric/Research/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.utils import diagonal_preprocessing
from data_summary_plots import genomic_distance_statistics

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--save_npy', action='store_true', help='true to save y as .npy')
    parser.add_argument('--ifile', default=osp.join('data_out','contacts.txt'), help='location of contact map')
    parser.add_argument('--odir', default='', help='path to output directory (default is wd)')

    args = parser.parse_args()
    return args

def plotContactMap(y, ofile = None, title = None, vmin = 0, vmax = 1, size_in = 6, minVal = None, maxVal = None, cmap = None):
    """
    Plotting function for contact maps.

    Inputs:
        y: contact map numpy array
        ofile: save location
        title: plot title
        vmax: maximum value for color bar, 'mean' to set as mean value
        size_in: size of figure x,y in inches
        minVal: values in y less than minVal are set to 0
        maxVal: values in y greater than maxVal are set to 0
    """
    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0,    'white'),
                                                  (1,    'red')], N=126)
    if minVal is not None:
        ind = y < minVal
        y[ind] = 0
    if maxVal is not None:
        ind = y > maxVal
        y[ind] = 0
    plt.figure(figsize = (size_in, size_in))
    if vmax == 'mean':
        vmax = np.mean(y)
    elif vmax == 'max':
        vmax = np.max(y)
    if vmin == 'min':
        vmin = np.min(y)
    ax = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap)
    if title is not None:
        plt.title(title, fontsize = 16)
    plt.tight_layout()
    if ofile is not None:
        plt.savefig(ofile)
    else:
        plt.show()
    plt.close()


def main():
    args = getArgs()
    if args.ifile.endswith('.txt'):
        y = np.loadtxt(args.ifile)[:args.m, :args.m]
    elif args.ifile.endswith('.npy'):
        y = np.load(args.ifile)[:args.m, :args.m]
    else:
        raise Exception("Unrecognized extension: {}".format(args.ifile))


    plotContactMap(y, ofile = 'y.png', vmax = 'mean')
    if args.save_npy:
        np.save(osp.join(args.odir, 'y.npy'), y.astype(np.int16))

        meanDist = genomic_distance_statistics(y)
        y_diag_instance = diagonal_preprocessing(y, meanDist)
        plotContactMap(y_diag_instance, ofile = osp.join(args.odir, 'y_diag.png'), vmax = 'max')
        np.save(osp.join(args.odir, 'y_diag.npy'), y_diag_instance)

if __name__ == '__main__':
    main()
