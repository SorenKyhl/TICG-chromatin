import os
import os.path as osp
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy
import numpy as np
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.xyz import (Q_similarity, calculate_rg, xyz_load,
                             xyz_to_contact_grid, xyz_to_distance, xyz_write)
from sklearn.cluster import AgglomerativeClustering

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_import_log

EXP_DIR = '/home/erschultz/Su2020/Bintu'
TICG_DIR = '/home/erschultz/Su2020/samples/sample1013_rescale1'

def calc_Q_arr(D, odir):
    Q_file = osp.join(odir, 'Q.npy')
    N = len(D)
    if osp.exists(Q_file):
        Q_arr = np.load(Q_file)
    else:
        Q_arr = np.zeros((N,N))
        for i in range(N):
            progress = np.round(i / N * 100, 1)
            if progress % 10 == 0:
                print(f'{progress}%')
            for j in range(i+1, N):
                Q_diff = Q_similarity(D[i], D[j], 165, diff = True)
                Q_arr[i,j] = Q_diff
                Q_arr[j,i] = Q_diff
        np.save(Q_file, Q_arr)

    return Q_arr

def cluster_and_plot(Q_arr, xyz, odir):
    k=2
    agg = AgglomerativeClustering(n_clusters = k, affinity='precomputed',
                                linkage = 'complete')
    agg.fit(Q_arr)
    print(Counter(agg.labels_))

    for cluster in range(k):
        inds = agg.labels_== cluster
        xyz_ind = xyz[inds]
        print(xyz_ind.shape)

        np.save(osp.join(odir, f'xyz_cluster{cluster}'), xyz_ind)
        xyz_write(xyz_ind, osp.join(odir, f'xyz_cluster{cluster}.xyz'), 'w')

        y_cluster = xyz_to_contact_grid(xyz_ind, 151.613)
        plot_matrix(y_cluster, osp.join(odir, f'y_cluster{cluster}.png'),
                    vmax = 'mean')

        rg = calculate_rg(xyz_ind, full_distribution = True)
        weights = np.ones_like(rg) / len(rg)
        plt.hist(rg, bins = 50, alpha = 0.5, label = f'cluster{cluster}')

    plt.legend()
    plt.ylabel('counts', fontsize=16)
    plt.xlabel(f'Rg', fontsize=16)
    plt.savefig(osp.join(odir, f'rg_distribution_clustered.png'))
    plt.close()

def experiment():
    xyz_file = osp.join(EXP_DIR, 'xyz2.npy')
    xyz1 = np.load(xyz_file)

    xyz_file = osp.join(EXP_DIR, 'K562/xyz2.npy')
    xyz2 = np.load(xyz_file)

    xyz = np.concatenate((xyz1, xyz2))
    N, m, _ = xyz.shape

    keep = []
    for i, xyz_i in enumerate(xyz):
        num_nans = np.sum(np.isnan(xyz_i[:, 0]))
        if num_nans / m < 0.1:
            keep.append(i)

    xyz_keep = xyz[keep]

    rg = calculate_rg(xyz_keep, full_distribution = True)
    plt.hist(rg, weights = np.ones_like(rg) / len(rg),
                                bins = 50)
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(f'Rg', fontsize=16)
    plt.savefig(osp.join(EXP_DIR, f'rg_distribution.png'))
    plt.close()

    print(f'# with Rg < 200: {np.sum(rg < 200)}')
    keep2 = rg > 200
    xyz_keep = xyz_keep[keep2]

    N = len(xyz_keep)
    print(f'Using {N} structures')

    # center structures (for visualization)
    for i in range(N):
        center = np.nanmean(xyz_keep[i], axis=0)
        xyz_keep[i] -= center

    # get distance matrices
    D1 = np.load(osp.join(EXP_DIR, 'dist2.npy'))
    D2 = np.load(osp.join(EXP_DIR, 'K562/dist2.npy'))
    D = np.concatenate((D1, D2))[keep][keep2]

    Q_arr = calc_Q_arr(D, EXP_DIR)

    cluster_and_plot(Q_arr, xyz_keep, EXP_DIR)

def TICG():
    me_dir = 'optimize_grid_b_200_v_8_spheroid_1.5-max_ent10_xyz/iteration20/'
    xyz_file = osp.join(TICG_DIR, me_dir, 'production_out/output.xyz')
    xyz = xyz_load(xyz_file, save = True)

    rg = calculate_rg(xyz, full_distribution = True)
    plt.hist(rg, weights = np.ones_like(rg) / len(rg),
                                bins = 50)
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(f'Rg', fontsize=16)
    plt.savefig(osp.join(TICG_DIR, f'rg_distribution.png'))
    plt.close()

    N = len(xyz)
    print(f'Using {N} structures')

    # crop to 28-30Mb
    result = load_import_log(TICG_DIR)
    start = int(result['start_mb'])
    resolution = result['resolution_mb']
    start_ind = int((28 - start) / resolution)
    end_ind = int((30 - start) / resolution)
    print(f'cropping to {start_ind, end_ind}')
    xyz = xyz[:, start_ind:end_ind]

    y = np.load(osp.join(TICG_DIR, 'y.npy'))
    y = y[start_ind:end_ind, start_ind:end_ind]
    plot_matrix(y, osp.join(TICG_DIR, 'y_28_30_mb.png'), vmax='mean')


    # center structures (for visualization)
    for i in range(N):
        center = np.nanmean(xyz[i], axis=0)
        xyz[i] -= center



    rg = calculate_rg(xyz, full_distribution = True)
    plt.hist(rg, weights = np.ones_like(rg) / len(rg),
                                bins = 50)
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(f'Rg', fontsize=16)
    plt.savefig(osp.join(TICG_DIR, f'rg_distribution.png'))
    plt.close()

    # get distance matrices
    D = xyz_to_distance(xyz)
    print(D.shape)

    Q_arr = calc_Q_arr(D, TICG_DIR)

    cluster_and_plot(Q_arr, xyz, TICG_DIR)


def TICG_2():
    me_dir = 'optimize_grid_b_200_v_8_spheroid_1.5-max_ent10_xyz/iteration20/'
    xyz_file = osp.join(TICG_DIR, me_dir, 'production_out/output.xyz')
    xyz = xyz_load(xyz_file, save = True)
    odir = osp.join(TICG_DIR, '25_27mb')
    if not osp.exists(odir):
        os.mkdir(odir)

    rg = calculate_rg(xyz, full_distribution = True)
    plt.hist(rg, weights = np.ones_like(rg) / len(rg),
                                bins = 50)
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(f'Rg', fontsize=16)
    plt.savefig(osp.join(odir, f'rg_distribution.png'))
    plt.close()

    N = len(xyz)
    print(f'Using {N} structures')

    # crop to 25-27Mb
    result = load_import_log(TICG_DIR)
    start = int(result['start_mb'])
    resolution = result['resolution_mb']
    start_ind = int((25 - start) / resolution)
    end_ind = int((27 - start) / resolution)
    print(f'cropping to {start_ind, end_ind}')
    xyz = xyz[:, start_ind:end_ind]

    y = np.load(osp.join(TICG_DIR, 'y.npy'))
    y = y[start_ind:end_ind, start_ind:end_ind]
    plot_matrix(y, osp.join(odir, 'y.png'), vmax='mean')

    # center structures (for visualization)
    for i in range(N):
        center = np.nanmean(xyz[i], axis=0)
        xyz[i] -= center

    rg = calculate_rg(xyz, full_distribution = True)
    plt.hist(rg, weights = np.ones_like(rg) / len(rg),
                                bins = 50)
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(f'Rg', fontsize=16)
    plt.savefig(osp.join(odir, f'rg_distribution.png'))
    plt.close()

    # get distance matrices
    D = xyz_to_distance(xyz)
    print(D.shape)

    Q_arr = calc_Q_arr(D, odir)

    cluster_and_plot(Q_arr, xyz, odir)


if __name__ == '__main__':
    experiment()
    TICG()
    # TICG_2()
