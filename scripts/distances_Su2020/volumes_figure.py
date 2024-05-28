import json
import math
import os.path as osp
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
from pylib.utils.load_utils import get_final_max_ent_folder, load_import_log
from pylib.utils.xyz import calculate_rg, xyz_load, xyz_to_distance
from scipy.spatial import ConvexHull
from su2020_analysis import get_dirs


def figure():
    label_fontsize=24
    tick_fontsize=18
    letter_fontsize=26

    dir = '/home/erschultz/Su2020/samples'
    odir = '/home/erschultz/TICG-chromatin/figures'
    s_dir = osp.join(dir, 'sample1013')
    result = load_import_log(s_dir)
    start = result['start']
    resolution = result['resolution']
    chrom = int(result['chrom'])
    genome = result['genome']

    if genome == 'hg38':
        if chrom == 21:
            exp_dir = osp.join(dir, 'sample1')
        elif chrom == 2:
            exp_dir = osp.join(dir, 'sample10')
        else:
            raise Exception(f'Unrecognized chrom: {chrom}')
    coords_file = osp.join(exp_dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)

    b = 180; phi = 0.008; ar=1.5
    GNN_ID = None
    max_ent_dir, gnn_dir = get_dirs(s_dir, GNN_ID, b, phi, ar)
    final_dir = get_final_max_ent_folder(max_ent_dir)
    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_max_ent = xyz_load(file, multiple_timesteps = True)
    if gnn_dir is not None and osp.exists(gnn_dir):
        file = osp.join(gnn_dir, 'production_out/output.xyz')
        xyz_gnn = xyz_load(file, multiple_timesteps = True)
    else:
        xyz_gnn = None

    N, m, _ = xyz_max_ent.shape
    i = coords_dict[f'chr{chrom}:{start}-{start+resolution}']
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz_exp = np.load(xyz_file)
    xyz_exp = xyz_exp[:, i:i+m, :]


    def get_vols(xyz):
        if xyz is None:
            return None
        print('xyz', xyz.shape, xyz.dtype)
        D = xyz_to_distance(xyz)
        num_cells, num_coords, _ = xyz.shape
        vols = np.zeros(num_cells)
        r_spheres = np.zeros(num_cells)
        for i in range(num_cells):
            xyz_i = xyz[i]
            xyz_i = xyz_i[~np.isnan(xyz_i)].reshape(-1, 3)
            points = len(xyz_i)
            if points < 100:
                print(f'Insufficient points for cell {i}')
                vols[i] = np.NaN
            else:
                try:
                    hull = ConvexHull(xyz_i)
                except Exception:
                    print(xyz[i])
                    print(i, xyz_i.shape)
                    raise
                vols[i] = hull.volume * 1e-9 # convert to um^3

        mean_vol = np.nanmean(vols)
        median_vol = np.nanmedian(vols)
        std_vol = np.nanstd(vols)
        max_vol = np.nanmax(vols)
        print(f'median vol {mean_vol} um^3')
        print(f'mean vol {median_vol} um^3')
        print(f'std vol {std_vol}')
        print(f'max vol {max_vol}')
        return vols

    N=200
    exp_hull = get_vols(xyz_exp[:N])
    max_ent_hull = get_vols(xyz_max_ent)
    gnn_hull = get_vols(xyz_gnn)

    sizes = [16, 32, 64, 128, 256, 512]
    log_labels = [i*resolution for i in sizes]
    ref_rgs = np.zeros((len(sizes), 2))
    max_ent_rgs = np.zeros((len(sizes), 2))
    gnn_rgs = np.zeros((len(sizes), 2))
    for i, size in enumerate(sizes):
        left = int(256 - size/2)
        right = int(256 + size/2)
        xyz_exp_size = xyz_exp[:, left:right, :]
        xyz_max_ent_size = xyz_max_ent[:, left:right, :]
        if xyz_gnn is not None:
            xyz_gnn_size = xyz_gnn[:, left:right, :]
            gnn_rgs[i] = calculate_rg(xyz_gnn_size)

        ref_rgs[i] = calculate_rg(xyz_exp_size, verbose = False)
        max_ent_rgs[i] = calculate_rg(xyz_max_ent_size)


    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(14)

    # volumes
    ax = axes[0]
    bin_width = 0.5
    for arr, label, c in zip([exp_hull, max_ent_hull, gnn_hull],
                            ['Experiment', 'Maximum Entropy', 'GNN'],
                            ['k', 'b', 'r']):
        if arr is None:
            continue
        ax.hist(arr, alpha=0.5, label = label, color = c,
                    weights = np.ones_like(arr) / len(arr),
                    bins = np.arange(math.floor(min(arr)),
                                math.ceil(max(arr)) + bin_width,
                                bin_width))

    ax.set_xlim(None, np.nanpercentile(exp_hull, 99))
    ax.set_xlabel(r'Convex Hull Volume $\mu$m$^3$', fontsize=label_fontsize)
    ax.set_ylabel('Probability', fontsize=label_fontsize)
    ax.legend(fontsize = tick_fontsize)

    # rg
    ax = axes[1]
    ax.errorbar(log_labels, ref_rgs[:, 0], ref_rgs[:, 1], color = 'k',
                label = 'Experiment')
    ax.errorbar(log_labels, max_ent_rgs[:, 0], max_ent_rgs[:, 1], color = 'b',
                label = 'Max Ent')
    if xyz_gnn is not None:
        ax.errorbar(log_labels, gnn_rgs[:, 0], gnn_rgs[:, 1], color = 'r',
                    label = 'GNN')

    X = np.linspace(log_labels[0], log_labels[-1], 100)
    Y = np.power(X, 1/3)
    Y = Y * ref_rgs[0, 0] / np.min(Y) * 1.1
    ax.plot(X, Y, label = '1/3', ls='dotted', color = 'gray')

    Y = np.power(X, 1/4)
    Y = Y * ref_rgs[0, 0] / np.min(Y) * 0.9
    ax.plot(X, Y, label = '1/4', ls='dashed', color = 'gray')

    ax.set_ylabel('Radius of Gyration', fontsize=label_fontsize)
    ax.set_xlabel('Domain Size (bp)', fontsize=label_fontsize)
    ax.legend(fontsize=tick_fontsize)
    ax.set_xscale('log')
    # ax.set_yscale('log')

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    for n, ax in enumerate(axes):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.savefig(osp.join(odir, 'volumes.png'))
    plt.close()

if __name__ == '__main__':
    figure()
