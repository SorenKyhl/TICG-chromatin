import copy
import math
import os
import os.path as osp

import imageio
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing

RED_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                         [(0,    'white'),
                                          (1,    'red')], N=126)
BLUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                         [(0,    'white'),
                                          (1,    'blue')], N=126)
BLUE_RED_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                         [(0, 'blue'),
                                         (0.5, 'white'),
                                          (1, 'red')], N=126)
RED_BLUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                         [(0, 'red'),
                                         (0.5, 'white'),
                                          (1, 'blue')], N=126)


def plot_matrix(arr, ofile=None, title=None, vmin=0, vmax='max',
                    size_in=6, minVal=None, maxVal=None, prcnt=False,
                    cmap=RED_CMAP, x_tick_locs=None, x_ticks=None,
                    y_tick_locs=None, y_ticks=None, triu=False, lines=[]):
    """
    Plotting function for 2D arrays.

    Inputs:
        arr: numpy array
        ofile: save location (None to show instead)
        title: plot title
        vmax: maximum value for color bar, 'mean' to set as mean value
        size_in: size of figure x,y in inches
        minVal: values in y less than minVal are set to 0
        maxVal: values in y greater than maxVal are set to 0
        triu: True to plot line y = -x splitting upper and lower triangle
        lines: draw black line across rows/cols in lines
    """
    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        pass
    elif cmap is None:
        cmap = RED_CMAP
    elif prcnt:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0,       'white'),
                                                  (0.25,    'orange'),
                                                  (0.5,     'red'),
                                                  (0.74,    'purple'),
                                                  (1,       'blue')], N=10)
    elif cmap == 'blue':
        cmap == BLUE_CMAP
    elif cmap == 'soren':
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom",
                                                    [(0, "white"),
                                                    (0.3, "white"),
                                                    (1, "#FF0000")], N=126)
    elif cmap.replace('-', '').lower() == 'bluered':
        if vmax == 'max' and vmin != 'center1':
            vmin = vmax = 'center'
        cmap = BLUE_RED_CMAP
    else:
        raise Exception(f'Invalid cmap: {cmap}')

    if len(arr.shape) == 4:
        N, C, H, W = arr.shape
        assert N == 1 and C == 1
        arr = arr.reshape(H,W)
    elif len(arr.shape) == 3:
        N, H, W = arr.shape
        assert N == 1
        y = arr.reshape(H,W)
    else:
        H, W = arr.shape

    if minVal is not None or maxVal is not None:
        arr = arr.copy() # prevent issues from reference type
    if minVal is not None:
        ind = arr < minVal
        arr[ind] = 0
    if maxVal is not None:
        ind = arr > maxVal
        arr[ind] = 0
    plt.figure(figsize = (size_in, size_in))

    # set min and max
    if vmin == 'center':
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        vmax = max(vmax, vmin * -1)
        vmin = vmax * -1
    elif vmin == 'center1':
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        d_vmin = 1-vmin
        d_vmax = vmax-1
        d = max(d_vmax, d_vmin)
        vmin = 1 - d
        vmax = 1 + d
    else:
        if vmin == 'min':
            vmin = np.nanpercentile(arr, 1)
            print(vmin)
            # uses 1st percentile instead of absolute min
        elif vmin == 'abs_min':
            vmin = np.min(arr)

        if vmax == 'mean':
            vmax = np.mean(arr)
        elif vmax == 'max':
            vmax = np.nanpercentile(arr, 99)
            # uses 99th percentile instead of absolute max
        elif vmax == 'abs_max':
            vmax = np.max(arr)

    ax = sns.heatmap(arr, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    if x_ticks is None:
        pass
    elif len(x_ticks) == 0:
        ax.axes.get_xaxis().set_visible(False)
    else:
        # ax.set_xticks([i+0.5 for i in range(W)])
        ax.set_xticks(x_tick_locs)
        ax.axes.set_xticklabels(x_ticks, rotation='horizontal')

    if y_ticks is None:
        pass
    elif len(y_ticks) == 0:
        ax.axes.get_yaxis().set_visible(False)
    else:
        # ax.set_yticks([i+0.5 for i in range(H)])
        ax.set_yticks(y_tick_locs)
        ax.axes.set_yticklabels(y_ticks, rotation='horizontal')
    ax.tick_params(axis='both', which='major', labelsize=10)

    if triu:
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*W, 0.01*H, 'Simulation', fontsize=16, ha='right', va='top')
        ax.text(0.01*W, 0.99*H, 'Experiment', fontsize=16)

    for line in lines:
        ax.axhline(line, color = 'k', lw=0.5)

    if title is not None:
        plt.title(title, fontsize = 16)
    plt.tight_layout()
    if ofile is not None:
        plt.savefig(ofile)
    else:
        plt.show()
    plt.close()

def plot_matrix_gif(arr, dir, ofile = None, title = None, vmin = 0, vmax = 1,
                    size_in = 6, minVal = None, maxVal = None, prcnt = False,
                    cmap = None, x_ticks = None, y_ticks = None):
    filenames = []
    for i in range(len(arr)):
        fname=osp.join(dir, f'{i}.png')
        filenames.append(fname)
        plot_matrix(arr[i,], fname, ofile, title, vmin, vmax, size_in, minVal,
                    maxVal, prcnt, cmap, x_ticks, y_ticks)

    # build gif
    # filenames = [osp.join(dir, f'ovito0{i}.png') for i in range(100, 900)]
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    imageio.mimsave(ofile, frames, format='GIF', fps=1)

    # remove files
    for filename in set(filenames):
        os.remove(filename)

def plot_mean_dist(meanDist, path, ofile, diag_chis_step, logx, ref,
                    ref_label = 'reference', label = '', color = 'blue',
                    title = None, ylabel='Contact Probability'):
    '''
    Inputs:
        meanDist (array): mean value along off-diagonals of contact map
        path (path): save path
        ofile (str): save file name
        diag_chis_step (array or None): diagonal chi parameter as function of d (None to skip)
        logx (bool): True to log-scale x axis
        ref (array): reference meanDist
        ref_label (str): label for legend
        label (str): label for legend
        color (str): color for meanDist
        title (str): plt title
        ylabel (str): y-axis label
    '''
    meanDist = meanDist.copy()
    if ref is not None:
        ref = ref.copy()

    fig, ax = plt.subplots()
    if ref is not None:
        ax.plot(ref, label = ref_label, color = 'k')
    ax.plot(meanDist, label = label, color = color)
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    if diag_chis_step is not None:
        ax2 = ax.twinx()
        ax2.plot(diag_chis_step, ls = '--', label = 'Parameters', color = color)
        ax2.set_ylabel('Diagonal Parameter', fontsize = 16)
        if logx:
            ax2.set_xscale('log')
        ax2.legend(loc='upper right')
    # else:
    #     x = np.arange(1, 10).astype(np.float64)
    #     ax.plot(x, np.power(x, -1)/4, color = 'grey', ls = '--', label='-1')
    #     x = np.arange(10, 100).astype(np.float64)
    #     ax.plot(x, np.power(x, -1.5), color = 'grey', ls = ':', label='-3/2')
    #     ax.legend(loc='upper right')

    ax.set_ylabel(ylabel, fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(osp.join(path, ofile))
    plt.close()

def plot_mean_vs_genomic_distance(y, path, ofile, diag_chis_step = None,
                                config = None, logx = False, ref = None,
                                ref_label = 'reference'):
    '''
    Wrapper for plot_mean_dist that takes contact map as input.

    Inputs:
        y: contact map
        path: save path
        ofile: save file name
        diag_chis_step: diagonal chi parameter as function of d (None to skip)
        config: config file (None to skip)
        logx: True to log-scale x axis
    '''
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob',
                                            zero_diag = False, zero_offset = 0)
    if ref is not None and ref.shape == y.shape:
        # ref is a contact map
        # else ref is already meanDist
        ref = DiagonalPreprocessing.genomic_distance_statistics(ref, 'prob',
                                                zero_diag = False, zero_offset = 0)
    if config is not None:
        diag_chis_step = calculate_diag_chi_step(config)

    plot_mean_dist(meanDist, path, ofile, diag_chis_step, logx, ref, ref_label)

    return meanDist


def test():
    y = np.random.normal(size=(512, 512))
    plot_matrix(y)

if __name__ == '__main__':
    test()
