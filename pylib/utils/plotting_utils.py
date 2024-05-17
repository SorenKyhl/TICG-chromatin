import copy
import math
import os
import os.path as osp
import sys

try:
    import cv2  # opencv-python
except ImportError as e:
    pass
import imageio
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_diag_chi_step

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
YELLOW_BLUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                         [(0, 'yellow'),
                                         (0.5, 'white'),
                                          (1, 'blue')], N=126)
RED_GREEN_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                         [(0, 'red'),
                                         (0.5, 'white'),
                                          (1, 'green')], N=126)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),cv2.INTER_NEAREST)

def plot_matrix(arr, ofile=None, title=None, vmin=0, vmax='max',
                size_in=6, minVal=None, maxVal=None, prcnt=False,
                cmap=RED_CMAP, x_tick_locs=None, x_ticks=None,
                y_tick_locs=None, y_ticks=None, triu=False, lines=[],
                percentile=1, use_cbar=True, loci=None, loci_size=20,
                border=False):
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
    elif isinstance(cmap, matplotlib.colors.ListedColormap):
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

    if border:
        plt.rcParams["axes.linewidth"] = 2

    # set up size
    if use_cbar:
        plt.figure(figsize = (size_in*1.2, size_in))
    else:
        plt.figure(figsize = (size_in, size_in))

    # set min and max
    if vmin == 'center':
        vmin = np.nanpercentile(arr, percentile)
        vmax = np.nanpercentile(arr, 100-percentile)
        vmax = max(vmax, vmin * -1)
        vmin = vmax * -1
    elif vmin == 'center1':
        vmin = np.nanpercentile(arr, percentile)
        vmax = np.nanpercentile(arr, 100-percentile)
        d_vmin = 1-vmin
        d_vmax = vmax-1
        d = max(d_vmax, d_vmin)
        vmin = 1 - d
        vmax = 1 + d
    else:
        if vmin == 'min':
            vmin = np.nanpercentile(arr, percentile)
            # uses 1st percentile instead of absolute min
        elif vmin == 'abs_min':
            vmin = np.min(arr)

        if vmax == 'mean':
            vmax = np.mean(arr)
        elif vmax == 'max':
            vmax = np.nanpercentile(arr, 100-percentile)
            # uses 99th percentile instead of absolute max
        elif vmax == 'abs_max':
            vmax = np.max(arr)

    ax = sns.heatmap(arr, linewidth = 0, vmin = vmin, vmax = vmax,
                    cmap = cmap, cbar = use_cbar)
    if use_cbar:
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
        ax.axhline(line + 0.5, color = 'k', lw=0.5)
        ax.axvline(line + 0.5, color = 'k', lw=0.5)

    if loci is not None:
        loci = [i + 0.5 for i in loci]
        ellipse = matplotlib.patches.Ellipse(loci, loci_size, loci_size, color='b', fill=False)
        ax.add_patch(ellipse)

    if border:
        sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)

    if title is not None:
        plt.title(title, fontsize = 16)
    plt.tight_layout()
    if ofile is not None:
        plt.savefig(ofile)
    else:
        plt.show()
    plt.close()

def plot_matrix_layout(rows, cols, ind, data_arr, val_arr=None, labels_arr=None,
                        cmap=RED_CMAP,
                        vmin=0, vmax=1, ofile=None, loci=None, loci_size=20):
    height = 6*rows; width = 6*cols
    width_ratios = [1]*cols+[0.08]
    fig, ax = plt.subplots(rows, cols+1,
                            gridspec_kw={'width_ratios':width_ratios})
    ax = np.atleast_2d(ax)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    row = 0; col=0
    if val_arr is None:
        val_arr = np.array([None]*len(data_arr))
    if labels_arr is None:
        labels_arr = np.array([None]*len(data_arr))
    if ind is None:
        ind = np.arange(0, len(data_arr), 1)
    for y, val, label in zip(data_arr[ind], val_arr[ind], labels_arr[ind]):
        if col == 0:
            s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap,
                ax = ax[row][col], cbar_ax = ax[row][-1])
        else:
            s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap,
                ax = ax[row][col], cbar = False)
        if val is None:
            s.set_title(f'{label}', fontsize = 16)
        else:
            s.set_title(f'{label}\n Score = {np.round(val, 1)}', fontsize = 16)
        s.set_xticks([])
        s.set_yticks([])

        col += 1
        if col == cols:
            col = 0
            row += 1

        if loci is not None:
            loci = [i + 0.5 for i in loci]
            ellipse = matplotlib.patches.Ellipse(loci, loci_size, loci_size, color='b', fill=False)
            s.add_patch(ellipse)

    plt.tight_layout()
    if ofile is None:
        plt.show()
    else:
        plt.savefig(ofile)
        plt.close()

def plot_matrix_gif(arr, dir, ofile=None, title=None, vmin=0, vmax=1,
                    size_in=6, minVal=None, maxVal=None, prcnt=False,
                    cmap=None, x_ticks=None, y_ticks=None):
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

def plot_mean_dist(meanDist, path, ofile, diag_chis_step, logx, logy=True,
                    ref=None, ref_label='reference', ref_color='k',
                    ref2=None, ref2_label='refernce 2', ref2_color='k',
                    label='', color='blue', title=None,
                    ylabel='Contact Probability'):
    '''
    Inputs:
        meanDist (array): mean value along off-diagonals of contact map
        path (path): save path
        ofile (str): save file name
        diag_chis_step (array or None): diagonal chi parameter as function of d
                                        (None to skip)
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
    if ref2 is not None:
        print('ref2', ref2[:5])
        ref2 = ref2.copy()

    fig, ax = plt.subplots()
    if ref is not None:
        ax.plot(ref, label = ref_label, color = ref_color)
    if ref2 is not None:
        print(ref2[:10])
        ax.plot(ref2, label = ref2_label, color = ref2_color)
    ax.plot(meanDist, label = label, color = color)
    ax.legend(loc='upper left')
    if logy:
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
    else:
        x = np.arange(1, 20).astype(np.float64)
        ax.plot(x, np.power(x, -1.5)/12, color = 'grey', ls = '--', label='-3/2')
        x = np.arange(1, 20).astype(np.float64)
        ax.plot(x, np.power(x, -1)/6, color = 'grey', ls = ':', label='-1')
        ax.legend(loc='upper right', fontsize=14)

    # if np.min(meanDist[meanDist != 0]) < 1e-6:
    #     plt.ylim(1e-6, 2)
    # else:
    #     plt.ylim(None, 2)

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

    plot_mean_dist(meanDist, path, ofile, diag_chis_step, logx, ref=ref,
                    ref_label=ref_label)

    return meanDist

def plot_diag_chi(config, path, ref = None, ref_label = '', logx = False,
                ofile = None, diag_chis_step = None, ylim = (None, None),
                title = None, label = ''):
    '''
    config: config file
    path: save file path
    ref: reference parameters
    ref_label: label for reference parameters
    '''
    if config is None:
        assert diag_chis_step is not None
    else:
        diag_chis_step = calculate_diag_chi_step(config)

    fig, ax = plt.subplots()
    ax.plot(diag_chis_step, color = 'k', label = label)
    ax.set_xlabel('Polymer Distance', fontsize = 16)
    ax.set_ylabel('Diagonal Parameter', fontsize = 16)
    if ref is not None:
        if isinstance(ref, str) and osp.exists(ref):
            ref = np.load(ref)

        if isinstance(ref, np.ndarray):
            ax.plot(ref, color = 'k', ls = '--', label = ref_label)
            if ref_label != '':
                plt.legend()

    ax.set_ylim(ylim[0], ylim[1])
    if logx:
        ax.set_xscale('log')
    if title is not None:
        plt.title(title)

    if ofile is None:
        if logx:
            ofile = osp.join(path, 'chi_diag_step_log.png')
        else:
            ofile = osp.join(path, 'chi_diag_step.png')
    else:
        ofile = osp.join(path, ofile)

    plt.savefig(ofile)
    plt.close()

    return diag_chis_step


### Functions for plotting sequences ###
def plot_seq_binary(seq, show=False, save=True, title=None, labels=None,
                    x_axis=True, ofile='seq.png', split=False):
    '''Plotting function for *non* mutually exclusive binary particle types'''
    m, k = seq.shape
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = cmap(ind)

    plt.figure(figsize=(6, 3))
    j = 0
    for i in range(k):
        c = colors[j]
        if split:
            j += i % 2
        else:
            j += 1
        x = np.argwhere(seq[:, i] == 1)
        if labels is None:
            label_i = i
        else:
            label_i = labels[i]
        plt.scatter(x, np.ones_like(x) * i * 0.2, label = label_i, color = c, s=3)

    ax = plt.gca()
    # ax.axes.get_yaxis().set_visible(False)
    if not x_axis:
        ax.axes.get_xaxis().set_visible(False)
    # else:
    #     ax.set_xticks(range(0, 1040, 40))
    #     ax.axes.set_xticklabels(labels = range(0, 1040, 40), rotation=-90)
    ax.set_yticks([i*0.2 for i in range(k)])
    ax.axes.set_yticklabels(labels = [f'Label {i}' for i in range(1,k+1)],
                            rotation='horizontal', fontsize=14)
    if title is not None:
        plt.title(title, fontsize=16)
    plt.xlabel('Distance', fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(ofile)
    if show:
        plt.show()
    plt.close()

def plot_seq_continuous(seq, show=False, save=True, title=None, ofile='seq.png',
                    split=False):
    m, k = seq.shape
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = cmap(ind)

    plt.figure(figsize=(6, 3))
    j=0
    for i in range(k):
        c = colors[j]
        if split:
            j += i % 2
        else:
            j += 1
        plt.plot(np.arange(0, m), seq[:, i], label = f'Label {i+1}', color = c)
        # i+1 to switch to 1-indexing

    ax = plt.gca()
    if title is not None:
        plt.title(title, fontsize=16)
    plt.legend(loc='upper right')
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('Label Value', fontsize=16)
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(ofile)
    plt.close()
