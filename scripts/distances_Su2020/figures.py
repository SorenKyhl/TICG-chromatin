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
import seaborn as sns
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import (BLUE_CMAP, BLUE_RED_CMAP,
                                        RED_BLUE_CMAP, RED_CMAP, plot_matrix,
                                        plot_mean_dist, rotate_bound)
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import pearson_round
from pylib.utils.xyz import xyz_load
from sklearn.decomposition import PCA

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import ArgparserConverter
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_import_log, load_Y)

sys.path.append('/home/erschultz/TICG-chromatin')
from scripts.distances_Su2020.su2020_analysis import (dist_distribution_a_b,
                                                      get_dirs, get_pcs,
                                                      load_exp_gnn_pca,
                                                      min_MSE,
                                                      rescale_mu_sigma)


def load_exp_gnn_pca_contact_maps(dir, GNN_ID=None, b=140, phi=0.03, ar=1.0):
    result = load_import_log(dir)
    start = result['start']
    resolution = result['resolution']
    chrom = int(result['chrom'])
    genome = result['genome']

    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi, ar)
    if osp.exists(max_ent_dir):
        final = get_final_max_ent_folder(max_ent_dir)
        y_pca, _ = load_Y(final)
        y_pca /= np.mean(np.diagonal(y_pca))
        m = len(y_pca)
    else:
        print(f'{max_ent_dir} does not exist')
        y_pca = None

    if GNN_ID is not None and osp.exists(gnn_dir):
        y_gnn, _ = load_Y(gnn_dir)
        y_gnn /= np.mean(np.diagonal(y_gnn))
        m = len(y_gnn)
    else:
        y_gnn = None

    y, _ = load_Y(dir)
    y /= np.mean(np.diagonal(y))

    return y, y_gnn, y_pca

def old_figure(sample, GNN_ID, bl=140, phi=0.03, ar=1.0):
    label_fontsize=24
    tick_fontsize=22
    letter_fontsize=26
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID, b=bl, phi=phi, ar=ar, mode='mean')
    D2 = np.load('/home/erschultz/Su2020/samples/sample1/dist2_mean.npy')
    nan_rows = np.isnan(D[0])
    D_no_nan = D[~nan_rows][:, ~nan_rows] # ignore nan_rows
    # mu_D_pca, sigma_D_pca, mu_D, sigma_D = rescale_mu_sigma(D, D_pca, True)
    # mu_D_gnn, sigma_D_gnn, _, _ = rescale_mu_sigma(D, D_gnn, True)
    # D_pca = rescale_mu_sigma(D, D_pca)
    # D_gnn = rescale_mu_sigma(D, D_gnn)
    # alpha_pca = min_MSE(D_no_nan, D_pca[~nan_rows][:, ~nan_rows])
    # alpha_gnn = min_MSE(D_no_nan, D_gnn[~nan_rows][:, ~nan_rows])
    alpha_pca = 1; alpha_gnn = 1
    D_pca = D_pca * alpha_pca
    if D_gnn is not None:
        D_gnn = D_gnn * alpha_gnn


    # compare PCs
    smooth = False; h = 1
    V_D = get_pcs(D, nan_rows, smooth=smooth, h = h)
    V_D_pca = get_pcs(D_pca, nan_rows, smooth=smooth, h = h)
    V_D_gnn = get_pcs(D_gnn, nan_rows, smooth=smooth, h = h)


    result = load_import_log(dir)
    start = result['start']
    end = result['end']
    start_mb = result['start_mb']
    end_mb = result['end_mb']
    chrom = int(result['chrom'])
    resolution = result['resolution']

    # distance distribution
    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    coords_a = f"chr{chrom}:15500001-15550001"
    coords_b = f"chr{chrom}:20000001-20050001"
    coords_a_label =  f"chr{chrom}:15.5 -15.55 Mb"
    coords_b_label = f"chr{chrom}:20-20.05 Mb"
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz = np.load(xyz_file)
    coords_file = osp.join(exp_dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)

    a = coords_dict[coords_a]
    b = coords_dict[coords_b]
    dist = dist_distribution_a_b(xyz, a, b)

    # shift ind such that start is at 0
    shift = coords_dict[f"chr{chrom}:{start}-{start+resolution}"]

    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, bl, phi, ar)
    final_dir = get_final_max_ent_folder(max_ent_dir)
    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_max_ent = xyz_load(file, multiple_timesteps = True) * alpha_pca
    dist_max_ent = dist_distribution_a_b(xyz_max_ent, a - shift, b - shift)
    # print(mu_D_pca, mu_D, sigma_D_pca, sigma_D)
    # dist_max_ent = (dist_max_ent - mu_D_pca)/sigma_D_pca * sigma_D + mu_D
    # print(dist_max_ent)

    if gnn_dir is not None and osp.exists(gnn_dir):
        file = osp.join(gnn_dir, 'production_out/output.xyz')
        print(file)
        xyz_gnn = xyz_load(file, multiple_timesteps = True) * alpha_gnn
        dist_gnn = dist_distribution_a_b(xyz_gnn, a - shift, b - shift)
        # dist_gnn = (dist_gnn - mu_D_gnn)/sigma_D_gnn * sigma_D + mu_D
    else:
        dist_gnn = None

    m = len(D[~nan_rows][:, ~nan_rows])
    all_labels = np.linspace(start_mb, end_mb, m)
    all_labels = np.round(all_labels, 0).astype(int)
    genome_ticks = [0, m-1]
    genome_labels = [f'{all_labels[i]}' for i in genome_ticks]

    ### combined figure ###
    print('---'*9)
    print('Starting Figure')
    plt.figure(figsize=(18, 10))
    ax1 = plt.subplot(2, 24, (1, 6))
    ax2 = plt.subplot(2, 24, (9, 14))
    ax_cb = plt.subplot(2, 48, 31)
    ax_legend = plt.subplot(2, 48, (35, 48))
    ax3 = plt.subplot(2, 24, (25, 30))
    ax4 = plt.subplot(2, 24, (33, 40)) # pc
    ax5 = plt.subplot(2, 12, (22, 24)) # dist a-b
    axes = [ax1, ax2]

    # plot dmaps
    vmin = np.nanpercentile(D_no_nan, 5)
    vmed = np.nanpercentile(D_no_nan, 50)
    vmax = np.nanpercentile(D_no_nan, 95)

    print(vmin, vmed, vmax)
    npixels = np.shape(D_no_nan)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)
    data = zip([D_gnn, D_pca], ['GNN', 'Max Ent'])
    for i, (D_sim, label) in enumerate(data):
        if D_sim is None:
            continue

        D_sim = D_sim[~nan_rows][:, ~nan_rows] # ignore nan_rows

        # make composite contact map
        composite = np.zeros((npixels, npixels))
        composite[indu] = D_sim[indu]
        composite[indl] = D_no_nan[indl]

        # get corr
        triu_ind = np.triu_indices(len(D_sim))
        corr = pearson_round(D_no_nan[triu_ind], D_sim[triu_ind], stat = 'nan_pearson')

        if i == 0:
            s = sns.heatmap(composite, linewidth = 0, vmin = vmin, vmax = vmax,
                            cmap = RED_BLUE_CMAP,
                            ax = axes[i], cbar = False)
        else:
            s = sns.heatmap(composite, linewidth = 0, vmin = vmin, vmax = vmax,
                            cmap = RED_BLUE_CMAP,
                            ax = axes[i], cbar_ax = ax_cb)
        s.axline((0,0), slope=1, color = 'k', lw=1)
        s.text(0.99*m, 0.01*m, label, fontsize=letter_fontsize, ha='right', va='top',
                weight = 'bold',
                path_effects=[pe.withStroke(linewidth=4, foreground="white")])
        s.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold',
                path_effects=[pe.withStroke(linewidth=4, foreground="white")])
        print(f'\nCorr(Exp, {label})={np.round(corr, 3)}')
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
                    fontsize = tick_fontsize)

        if i > 0:
            s.set_yticks([])
        else:
            s.set_yticks(genome_ticks, labels = genome_labels,
                        fontsize = tick_fontsize)

    ax_cb.tick_params(labelsize=tick_fontsize)

    # plot scaling
    m = len(D)
    log_labels = np.linspace(0, resolution*(m-1), m)
    # print('h1204', log_labels.shape)
    data = zip([D, D_pca, D_gnn], ['Experiment', 'Max Ent', 'GNN'], ['k', 'b', 'r'])
    for D_i, label, color in data:
        # print(label)
        if D_i is not None:
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_i, 'freq')
            nan_rows = np.isnan(meanDist)
            # print(meanDist[:10], meanDist.shape)
            ax3.plot(log_labels[~nan_rows], meanDist[~nan_rows], label = label,
                        color = color)

    meanDist = DiagonalPreprocessing.genomic_distance_statistics(D2, 'freq')
    log_labels = np.linspace(0, 30000*(len(meanDist)-1), len(meanDist))
    nan_rows = np.isnan(meanDist)
    print(meanDist[:10], meanDist.shape)
    ax3.plot(log_labels[~nan_rows], meanDist[~nan_rows], label = 'Experiment2',
                color = 'k', ls=':')

    ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax3.set_ylabel('Distance (nm)', fontsize = label_fontsize)
    ax3.set_xlabel('Genomic Separation (bp)', fontsize = label_fontsize)
    ax3.set_xscale('log')
    # ax3.set_yscale('log')
    # X = np.arange(1*10**5, 9*10**6, resolution)
    # A = .001/resolution
    # Y = A*np.power(X, 1/2) + 200
    # ax3.plot(X, Y, ls='dashed', color = 'grey')
    # ax3.legend(fontsize=legend_fontsize)

    # plot pcs
    ax4.plot(V_D[0], label = 'Experiment', color = 'k')
    if V_D_pca is not None:
        V_D_pca[0] *= np.sign(pearson_round(V_D[0], V_D_pca[0]))
        ax4.plot(V_D_pca[0], label = f'Maximum Entropy', color = 'blue')
        print(f'Max Ent (r={pearson_round(V_D[0], V_D_pca[0])})')
    if V_D_gnn is not None:
        V_D_gnn[0] *= np.sign(pearson_round(V_D[0], V_D_gnn[0]))
        ax4.plot(V_D_gnn[0], label = f'GNN', color = 'red')
        print(f'GNN (r={pearson_round(V_D[0], V_D_gnn[0])})')
    ax4.set_ylabel('PC 1', fontsize=label_fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax4.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
    ax4.set_yticks([])
    ax4.set_xlabel('Genomic Distance (Mb)', fontsize=label_fontsize)
    # ax4.legend(fontsize=legend_fontsize)

    # plot a-b distribution
    bin_width = 50
    arrs = [dist[~np.isnan(dist)], dist_max_ent, dist_gnn]
    labels = ['Experiment', 'Maximum Entropy', 'GNN']
    colors = ['k', 'b', 'r']
    data = zip(arrs, labels, colors)
    for arr, label, color in data:
        if arr is None:
            continue
        ax5.hist(arr, label = label, alpha = 0.5, color = color,
                    weights = np.ones_like(arr) / len(arr),
                    bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width))
    ax5.set_ylabel('Probability', fontsize=label_fontsize)
    ax5.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # ax5.legend(fontsize=legend_fontsize)
    ax5.set_xlabel('Spatial Distance (nm)', fontsize=label_fontsize)
    ax5.set_xlim(None, 3000)
    ax5.set_yticks([])
    print(f'Distance between\n{coords_a_label} and {coords_b_label}')

    h, l = ax4.get_legend_handles_labels()
    ax_legend.legend(h, l, borderaxespad=0, fontsize=label_fontsize, loc='center')
    ax_legend.axis("off")

    for n, ax in enumerate([ax1, ax3, ax4, ax5]):
        ax.text(-0.0, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')


    plt.subplots_adjust(bottom=0.1, top = 0.95, left = 0.1, right = 0.95,
                    hspace = 0.25, wspace = 0.25)
    plt.tight_layout()
    plt.savefig('/home/erschultz/TICG-chromatin/figures/distances.png')
    plt.close()

def new_figure(sample, GNN_ID, bl=140, phi=0.03):
    label_fontsize=24
    tick_fontsize=22
    letter_fontsize=26
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    D, D_gnn, _ = load_exp_gnn_pca(dir, GNN_ID, b=bl, phi=phi)
    nan_rows = np.isnan(D[0])

    # compare PCs
    smooth = False; h = 1
    V_D = get_pcs(D, nan_rows, smooth=smooth, h = h)
    V_D_gnn = get_pcs(D_gnn, nan_rows, smooth=smooth, h = h)

    result = load_import_log(dir)
    start = result['start']
    end = result['end']
    start_mb = result['start_mb']
    end_mb = result['end_mb']
    chrom = int(result['chrom'])
    resolution = result['resolution']

    # distance distribution
    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    coords_a = f"chr{chrom}:15500001-15550001"
    coords_b = f"chr{chrom}:20000001-20050001"
    coords_a_label =  f"chr{chrom}:15.5 -15.55 Mb"
    coords_b_label = f"chr{chrom}:20-20.05 Mb"
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz = np.load(xyz_file)
    coords_file = osp.join(exp_dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)

    a = coords_dict[coords_a]
    b = coords_dict[coords_b]
    dist = dist_distribution_a_b(xyz, a, b)

    # shift ind such that start is at 0
    shift = coords_dict[f"chr{chrom}:{start}-{start+resolution}"]

    _, gnn_dir = get_dirs(dir, GNN_ID, bl, phi)
    if gnn_dir is not None and osp.exists(gnn_dir):
        file = osp.join(gnn_dir, 'production_out/output.xyz')
        print(file)
        xyz_gnn = xyz_load(file, multiple_timesteps = True)
        dist_gnn = dist_distribution_a_b(xyz_gnn, a - shift, b - shift)
    else:
        dist_gnn = None

    m = len(D[~nan_rows][:, ~nan_rows])
    all_labels = np.linspace(start_mb, end_mb, m)
    all_labels = np.round(all_labels, 0).astype(int)
    genome_ticks = [0, m-1]
    genome_labels = [f'{all_labels[i]}' for i in genome_ticks]

    ### combined figure ###
    print('---'*9)
    print('Starting Figure')
    plt.figure(figsize=(18, 6))
    ax3 = plt.subplot(1, 24, (1, 6))
    ax4 = plt.subplot(1, 24, (8, 16)) # pc
    ax5 = plt.subplot(1, 24, (18, 24)) # dist a-b

    # plot scaling
    m = len(D)
    log_labels = np.linspace(0, resolution*(m-1), m)
    # print('h1204', log_labels.shape)
    data = zip([D, D_gnn], ['Experiment', 'GNN'], ['k','r'])
    for D_i, label, color in data:
        # print(label)
        if D_i is not None:
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_i, 'freq')
            nan_rows = np.isnan(meanDist)
            # print(meanDist[:10], meanDist.shape)
            ax3.plot(log_labels[~nan_rows], meanDist[~nan_rows], label = label, color = color)
    ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax3.set_ylabel('Distance (nm)', fontsize = label_fontsize)
    ax3.set_xlabel('Genomic Separation (bp)', fontsize = label_fontsize)
    ax3.set_xscale('log')
    # ax3.set_yscale('log')
    # X = np.arange(1*10**5, 9*10**6, resolution)
    # A = .001/resolution
    # Y = A*np.power(X, 1/2) + 200
    # ax3.plot(X, Y, ls='dashed', color = 'grey')
    # ax3.legend(fontsize=legend_fontsize)

    # plot pcs
    ax4.plot(V_D[0], label = 'Experiment', color = 'k')
    if V_D_gnn is not None:
        V_D_gnn[0] *= np.sign(pearson_round(V_D[0], V_D_gnn[0]))
        ax4.plot(V_D_gnn[0], label = f'GNN', color = 'red')
        print(f'GNN (r={pearson_round(V_D[0], V_D_gnn[0])})')
    ax4.set_ylabel('PC 1', fontsize=label_fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax4.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
    ax4.set_yticks([])
    ax4.set_xlabel('Genomic Distance (Mb)', fontsize=label_fontsize)
    # ax4.legend(fontsize=legend_fontsize)

    # plot a-b distribution
    bin_width = 50
    arrs = [dist[~np.isnan(dist)], dist_gnn]
    labels = ['Experiment', 'GNN']
    colors = ['k', 'r']
    data = zip(arrs, labels, colors)
    for arr, label, color in data:
        if arr is None:
            continue
        ax5.hist(arr, label = label, alpha = 0.5, color = color,
                    weights = np.ones_like(arr) / len(arr),
                    bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width))
    ax5.set_ylabel('Probability', fontsize=label_fontsize)
    ax5.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # ax5.legend(fontsize=legend_fontsize)
    ax5.set_xlabel('Spatial Distance (nm)', fontsize=label_fontsize)
    ax5.set_xlim(None, 3000)
    ax5.set_yticks([])
    print(f'Distance between\n{coords_a_label} and {coords_b_label}')

    ax4.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center',
                fontsize = label_fontsize,
                borderaxespad=0, ncol = 3)

    for n, ax in enumerate([ax3, ax4, ax5]):
        ax.text(-0.0, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')


    plt.subplots_adjust(bottom=0.275, top = 0.9, left = 0.15, right = 0.95,
                    hspace = 0.25, wspace = 0.45)
    plt.tight_layout()
    plt.savefig('/home/erschultz/TICG-chromatin/figures/distances.png')
    plt.close()

def supp_figure(sample, GNN_ID, bl, phi, ar):
    label_fontsize=24
    tick_fontsize=22
    letter_fontsize=26
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    y, y_gnn, y_pca = load_exp_gnn_pca_contact_maps(dir, GNN_ID, b=bl, phi=phi, ar=ar)
    m = len(y)

    ### supp figure with hicmaps ###
    print('---'*9)
    print('Starting Supp Figure')
    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(6.15)
    fig.set_figwidth(12)

    npixels = np.shape(y)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)
    composites = []
    for y_i in [y_gnn, y_pca]:
        # make composite contact map
        composite = np.zeros((npixels, npixels))
        composite[indu] = y_i[indu]
        composite[indl] = y[indl]
        composites.append(composite)

    # plot cmaps
    arr = np.array(composites)
    vmax = np.mean(arr)
    data = zip(composites, ['GNN', 'Max Ent'])
    for i, (composite, label) in enumerate(data):
        print(i)
        ax = axes[i]
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = ax, cbar = False)
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*m, 0.01*m, label, fontsize=letter_fontsize, ha='right', va='top',
                weight='bold')
        ax.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold')
        # s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
        # s.set_yticks(genome_ticks, labels = genome_labels)
        s.set_xticks([])
        s.set_yticks([])
        s.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    for n, ax in enumerate(axes):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.savefig('/home/erschultz/TICG-chromatin/figures/distances_hic.png')
    plt.close()



if __name__ == '__main__':
    old_figure(1004, None, bl=180, phi=0.01, ar=2.0)
    # supp_figure(1004, None, bl=180, phi=0.01, ar=2.0)
