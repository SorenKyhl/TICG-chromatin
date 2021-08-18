import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--save_npy', action='store_true', help='true to save y as .npy')
    parser.add_argument('--ifile', help='location of contact map')

    args = parser.parse_args()

    if args.ifile is None:
        args.ifile = osp.join('data_out','contacts.txt')
    return args

def plotContactMap(y, ofile = None, title = None, vmin = 0, vmax = 1, size_in = 6, minVal = None, maxVal = None):
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
        np.save('y.npy', y.astype(np.int16))

if __name__ == '__main__':
    main()
