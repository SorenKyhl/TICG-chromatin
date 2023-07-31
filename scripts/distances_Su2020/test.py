import csv
import json
import math
import os
import os.path as osp
import string
import sys

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import seaborn as sns
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import (BLUE_CMAP, BLUE_RED_CMAP,
                                        RED_BLUE_CMAP, plot_matrix,
                                        plot_mean_dist, rotate_bound)
from pylib.utils.similarity_measures import SCC
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import ArgparserConverter
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_import_log, load_Y)
from sequences_to_contact_maps.scripts.utils import (calc_dist_strat_corr,
                                                     nan_pearsonr,
                                                     pearson_round)
from sequences_to_contact_maps.scripts.xyz_utils import (calculate_rg,
                                                         xyz_load,
                                                         xyz_to_distance,
                                                         xyz_write)


def test_pcs():
    dir = '/home/erschultz/Su2020/samples'
    for f in ['sample10']:
        fdir = osp.join(dir, f)
        D_mean = np.load(osp.join(fdir, 'dist_mean.npy'))
        D_med = np.load(osp.join(fdir, 'dist_median.npy'))
        D_prox = np.load(osp.join(fdir, 'dist_proximity.npy'))
        nan_rows = np.isnan(D_mean[0])

        V_D_mean = get_pcs(D_mean, nan_rows)
        V_D_med = get_pcs(D_med, nan_rows)
        V_D_prox = get_pcs(D_prox, nan_rows)

        rows = 2; cols = 1
        row = 0; col = 0
        fig, ax = plt.subplots(rows, cols)
        fig.set_figheight(12)
        fig.set_figwidth(16)
        for i in range(rows*cols):
            ax[row].plot(V_D_mean[i], label = 'mean')
            ax[row].plot(V_D_med[i], label = 'median')
            ax[row].plot(V_D_prox[i], label = 'prox')
            ax[row].set_title(f'PC {i+1}')
            ax[row].legend(fontsize=16)

            row += 1
        plt.savefig(osp.join(fdir, 'pc_D.png'))
        plt.close()

def hg19_vs_hg38():
    dir = '/home/erschultz/Su2020/samples'
    fig, ax = plt.subplots()

    for s in [1002, 1003]:
        s_dir = osp.join(dir, f'sample{s}')
        y = np.load(osp.join(s_dir, 'y.npy'))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        print(meanDist[:10])
        ax.plot(meanDist, label = s)

    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    hg19_vs_hg38()
