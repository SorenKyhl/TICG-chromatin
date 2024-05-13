import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums, zscore
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from utils import get_samples, loci_to_coords

from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import (BLUE_RED_CMAP, RED_BLUE_CMAP, RED_CMAP,
                                        plot_matrix, plot_matrix_layout)
from pylib.utils.utils import load_import_log
from pylib.utils.xyz import calculate_rg, xyz_load, xyz_to_distance

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y


def plot_matrix_layout_zoom(inds, y_arr, cell_lines, loci,
                            odir, ofname, center = False, cmap=None):
    if center:
        vmin = np.nanpercentile(y_arr, 1)
        vmax = np.nanpercentile(y_arr, 100-1)
        d_vmin = 1-vmin
        d_vmax = vmax-1
        d = max(d_vmax, d_vmin)
        vmin = 1 - d
        vmax = 1 + d
        if cmap is None:
            cmap = BLUE_RED_CMAP
    else:
        vmin = 0
        vmax = np.mean(y_arr)
        if cmap is None:
            cmap = RED_CMAP
    N = len(inds)

    plot_matrix_layout(1, N, inds, y_arr, None, cell_lines,
                        vmin = vmin, vmax = vmax, cmap = cmap,
                        ofile = osp.join(odir, f'{ofname}.png'), loci = loci)

    w = 30
    left = np.min(loci) - w
    right = np.max(loci) + w
    new_loci = [i - left for i in loci]
    y_arr_zoom = y_arr[:, left:right, left:right]
    plot_matrix_layout(1, N, inds, y_arr_zoom, None, cell_lines,
                        vmin = vmin, vmax = vmax, cmap = cmap,
                        ofile = osp.join(odir, f'{ofname}_zoom.png'),
                        loci = new_loci)

    if N == 2:
        diff_zoom = y_arr_zoom[inds[0]] - y_arr_zoom[inds[1]]
        plot_matrix(diff_zoom, osp.join(odir, f'{ofname}_diff_zoom'), vmin = 'center',
                    cmap = RED_BLUE_CMAP, loci = new_loci)

def main():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures/DIs'
    if not osp.exists(odir):
        os.mkdir(odir)
    xyz_list = []
    D_list = []
    m=512
    samples, cell_lines = get_samples('dataset_12_06_23')
    samples = np.array(samples)
    cell_lines = np.array(cell_lines)
    gnn_root = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN631_xyz'
    N = len(samples)
    y_arr = np.zeros((N, m, m))
    y_diag_arr = np.zeros((N, m, m))
    for i, sample in enumerate(samples):
        # print(sample)
        s_dir = osp.join(dir, f'sample{sample}')

        y, y_diag = load_Y(s_dir)
        y_arr[i] = y
        y_diag_arr[i] = y_diag

        xyz_file = osp.join(s_dir, gnn_root, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps = True, N_min = 10,
                        verbose = False)
        n = len(xyz)
        xyz_list.append(xyz)

        D_file = osp.join(s_dir, 'D.npy')
        if osp.exists(D_file):
            D = np.load(D_file)
        else:
            D = xyz_to_distance(xyz)
            np.save(D_file, D)
        D_list.append(D)


    D_arr = np.array(D_list)
    D_mean_arr = np.mean(D_arr, axis=1)

    import_log = load_import_log(s_dir)

    # calc pvals for interactions up to distance K
    K = 40 # 40 = 2 Mb at 50kb resolution
    inds = [0, 1]
    all_pvals = []
    all_loci = []
    for k in range(1, K, 1):
        print(k)
        data = np.zeros((N, n, m-k))
        for i, D in enumerate(D_list):
            diag = np.diagonal(D, k, axis1=1, axis2=2)
            data[i] = diag

        loci_i = np.arange(0, m-k, 1)
        loci_j = np.arange(k, m, 1)
        loci = np.stack((loci_i, loci_j))

        # determine which tests to consider based on mean distance
        # this helps with multiple hypothesis testing
        data_mean = np.mean(data, axis=1) # N, m-k
        # print(np.percentile(data_mean, [10, 50, 90]))
        where = data_mean < 600 # N, m-k
        where = np.sum(where, axis=0).astype(bool) # m-k

        # filter loci based on where
        loci = loci[:, where]
        loci = loci.T.tolist()

        # z score normalize
        # data_zscore = zscore(data, axis = None, ddof=1) # N, n, m-k

        for i in inds:
            for j in inds:
                if i > j:
                    data_i = data[i, :, where].T
                    data_j = data[j, :, where].T
                    stats, pvals = ranksums(data_i, data_j,
                                            axis=0) # m-k[where]
                    all_pvals.extend(pvals)
                    all_loci.extend(loci)

    # FDR BH procedure
    rejects, pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.001, method='fdr_bh')
    print(f'{np.sum(rejects)} rejects out of {len(pvals_corrected)} tests')

    # plot reject_matrix showing loci of all rejects
    reject_matrix = np.zeros((m,m))
    for reject, loci in zip(rejects, all_loci):
        if reject:
            i, j = loci
            reject_matrix[i, j] = 1
            reject_matrix[j, i] = 1
    plot_matrix(reject_matrix, osp.join(odir, 'rejects.png'), vmin=0, vmax=1,
                use_cbar = False)

    diff = y_arr[inds[0]] - y_arr[inds[1]]
    plot_matrix(diff, osp.join(odir, 'hic_diff'), vmin = 'center',
                cmap = RED_BLUE_CMAP)
    diff = y_diag_arr[inds[0]] - y_diag_arr[inds[1]]
    plot_matrix(diff, osp.join(odir, 'hic_diag_diff'), vmin = 'center',
                cmap = RED_BLUE_CMAP)

    # report out top tests
    top = np.argsort(pvals_corrected)[:1]
    plotted = set()
    for i in top:
        pval = pvals_corrected[i]
        loci = all_loci[i]
        coords = loci_to_coords(loci, import_log)
        if loci[0] not in plotted:
            print(i, pval, loci, coords)
            plotted.add(loci[0])
            plot_matrix_layout_zoom(inds, y_arr, cell_lines,
                                    loci, odir, f'loci_{i}_hic')
            plot_matrix_layout_zoom(inds, y_diag_arr, cell_lines,
                                    loci, odir, f'loci_{i}_hic_diag', center = True)
            plot_matrix_layout_zoom(inds, D_mean_arr, cell_lines,
                                    loci, odir, f'loci_{i}_dist', cmap = RED_BLUE_CMAP)

            for ind in inds:
                # compare distance distributions w/ histogram
                arr = D_list[ind][:, loci[0], loci[1]]
                n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = 50, alpha = 0.5, label = cell_lines[ind])
                plt.legend(title = 'Cell Line')
            # stat, pval = ranksums(D_list[0][:, loci[0], loci[1]], D_list[1][:, loci[0], loci[1]])
            plt.ylabel('probability', fontsize=16)
            plt.xlabel('distance', fontsize=16)
            plt.savefig(osp.join(odir, f'loci_{i}_{cell_lines[inds[0]]}_{cell_lines[inds[1]]}.png'))
            # plt.show()
            plt.close()


def distance_distributions():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures'
    xyz_list = []
    D_list = []
    m=512
    samples, cell_lines = get_samples('dataset_12_06_23')
    samples = np.array(samples)
    cell_lines = np.array(cell_lines)
    gnn_root = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN631_xyz'
    N = len(samples)
    y_arr = np.zeros((N, m, m))
    y_diag_arr = np.zeros((N, m, m))
    for i, sample in enumerate(samples):
        # print(sample)
        s_dir = osp.join(dir, f'sample{sample}')

        y, y_diag = load_Y(s_dir)
        y_arr[i] = y
        y_diag_arr[i] = y_diag

        xyz_file = osp.join(s_dir, gnn_root, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps = True, N_min = 10,
                        verbose = False)
        n = len(xyz)
        xyz_list.append(xyz)

        D_file = osp.join(s_dir, 'D.npy')
        if osp.exists(D_file):
            D = np.load(D_file)
        else:
            D = xyz_to_distance(xyz)
            np.save(D_file, D)
        D_list.append(D)

        D_mean = np.mean(D, axis=0)

        meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_mean)
        plt.plot(meanDist, label = cell_lines[i])
    plt.legend()
    plt.xscale('log')
    plt.ylabel('3D distance', fontsize=16)
    plt.xlabel('1D distance', fontsize=16)
    plt.savefig(osp.join(odir, f'distance_scaling.png'))
    plt.close()


    inds = [0,2]
    # compare distribution over all distances w/ histogram
    for ind in inds:
        cell_line = cell_lines[ind]
        arr = D_list[ind].flatten()
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                bins = 50, alpha = 0.5, label = cell_line)
    plt.legend(title = 'Cell Line')

    # stat, pval = ranksums(D_list[0][:, loci[0], loci[1]], D_list[1][:, loci[0], loci[1]])
    plt.ylabel('probability', fontsize=16)
    plt.xlabel('distance', fontsize=16)
    plt.savefig(osp.join(odir, f'all_loci_{cell_lines[inds[0]]}_{cell_lines[inds[1]]}.png'))
    # plt.show()
    plt.close()

if __name__ == '__main__':
    main()
    # distance_distributions()
    # test_index()
    # z_score_test()
