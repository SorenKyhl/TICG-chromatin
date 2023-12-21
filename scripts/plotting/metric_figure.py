import os.path as osp
import string
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as ss
import seaborn as sns
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import (calculate_all_energy, calculate_D,
                                      calculate_diag_chi_step, calculate_L,
                                      calculate_S)
from pylib.utils.plotting_utils import RED_CMAP, rotate_bound
from pylib.utils.similarity_measures import SCC, hic_spector
from pylib.utils.utils import pearson_round
from pylib.utils.xyz import xyz_load, xyz_write

sys.path.append('/home/erschultz/TICG-chromatin/scripts')
from data_generation.modify_maxent import get_samples
from distances_Su2020.utils import plot_diagonal
from makeLatexTable import *

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (
    get_converged_max_ent_folder, get_final_max_ent_folder, load_import_log,
    load_L)


def metric_figure():
    test=False
    label_fontsize=22
    tick_fontsize=18
    letter_fontsize=26
    dataset = 'dataset_12_06_23'; GNN_ID = 631
    # dataset = 'dataset_04_05_23'; sample = 1001; GN_ID = 407
    # dataset = 'dataset_04_05_23'; sample = 1001; GNN_ID = 423
    samples, _ = get_samples(dataset, test=True, filter_cell_lines=['imr90'])
    N=4
    samples_list = samples[:N]
    print(f'Samples: {samples_list}, len={len(samples_list)}')
    k=10
    grid_root = 'optimize_grid_b_200_v_8_spheroid_1.5'
    def get_dirs(sample_dir):
        grid_dir = osp.join(sample_dir, grid_root)
        max_ent_dir = f'{grid_dir}-max_ent{k}'
        gnn_dir = f'{grid_dir}-GNN{GNN_ID}'

        return max_ent_dir, gnn_dir


    def get_y(sample_dir):
        max_ent_dir, gnn_dir = get_dirs(sample_dir)
        y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
        y /= np.mean(np.diagonal(y))

        final = get_final_max_ent_folder(max_ent_dir, 'normal')
        y_pca = np.load(osp.join(final, 'y.npy')).astype(np.float64)
        y_pca /= np.mean(np.diagonal(y_pca))

        y_gnn = np.load(osp.join(gnn_dir, 'y.npy')).astype(np.float64)
        y_gnn /= np.mean(np.diagonal(y_gnn))

        return y, y_pca, y_gnn


    scc = SCC(h=5, K=100)

    print('---'*9)
    print('Starting Figure')
    fig = plt.figure(figsize=(22, 12.5))
    ax1 = plt.subplot(2, 17, (1, 4))
    ax2 = plt.subplot(2, 17, (5, 8))
    ax3 = plt.subplot(2, 17, (10, 13))
    ax4 = plt.subplot(2, 17, (14, 17))
    ax5 = plt.subplot(2, 17, (18, 21))
    ax6 = plt.subplot(2, 17, (22, 25))
    ax7 = plt.subplot(2, 17, (27, 30))
    ax8 = plt.subplot(2, 17, (31, 34))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    ax_i = 0
    ellipses = [(375, 315, 50, 50), (300, 90, 240, 110),
                (370, 275, 80, 100), (280, 160, 160, 80)]

    for sample, (x, y, rx, ry) in zip(samples_list, ellipses):
        sample_dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        max_ent_dir, gnn_dir = get_dirs(sample_dir)
        y_gt, y_pca, y_gnn = get_y(sample_dir)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_gt)
        y_gt_diag = DiagonalPreprocessing.process(y_gt, meanDist)
        m = len(y_gt)

        result = load_import_log(sample_dir)
        start = result['start_mb']
        end = result['end_mb']
        chrom = result['chrom']
        resolution = result['resolution']
        resolution_mb = result['resolution_mb']
        all_labels = np.linspace(start, end, len(y_gt))
        all_labels = np.round(all_labels, 0).astype(int)
        # genome_ticks = [0, len(y)//3, 2*len(y)//3, len(y)-1]
        genome_ticks = [0, len(y_gt)-1]
        genome_labels = [f'{all_labels[i]}' for i in genome_ticks]

        vmin = 0
        vmax = np.mean(y_gt)
        npixels = np.shape(y_gt)[0]
        indu = np.triu_indices(npixels)
        indl = np.tril_indices(npixels)
        data = zip([y_gnn, y_pca], ['GNN', 'Max Ent'])
        scc = SCC(h=5, K=100)
        for i, (y_sim, label) in enumerate(data):
            # make composite contact map
            composite = np.zeros((npixels, npixels))
            composite[indu] = y_sim[indu]
            composite[indl] = y_gt[indl]

            s = sns.heatmap(composite, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                            ax = axes[ax_i], cbar = None)
            ax_i += 1
            if i == 0:
                s.set_yticks(genome_ticks, labels = genome_labels,
                        fontsize = tick_fontsize)
            else:
                s.set_yticks([])
            s.set_xticks([])

            pcs_gt = epilib.get_pcs(epilib.get_oe(y_gt), 12).T

            pcs_sim = epilib.get_pcs(epilib.get_oe(y_sim), 12).T
            pearson_pc_1, _ = pearsonr(pcs_sim[0], pcs_gt[0])
            pearson_pc_1 *= np.sign(pearson_pc_1) # ensure positive pearson
            assert pearson_pc_1 > 0
            pearson_pc_1 = np.round(pearson_pc_1, 3)

            y_diag_sim = epilib.get_oe(y_sim)
            rmse_y_tilde = mean_squared_error(y_gt_diag, y_diag_sim, squared=False)
            rmse_y_tilde = np.round(rmse_y_tilde, 3)

            scc_var = scc.scc(y_gt, y_sim, var_stabilized = True)
            scc_var = np.round(scc_var, 3)
            
            corr_spector = hic_spector(y_gt, y_sim, 10)
            corr_spector = np.round(corr_spector, 3)

            title = f'SCC={scc_var}\nHiC-Spector={corr_spector}\nCorr PC1('+r'$\tilde{H}$)'+f'={pearson_pc_1}\n'+r'RMSE($\tilde{H}$)'+f'={rmse_y_tilde}'
            s.set_title(title, fontsize = 16, loc='left')


            s.axline((0,0), slope=1, color = 'k', lw=1)
            s.text(0.99*m, -0.08*m, label, fontsize=label_fontsize, ha='right', va='top', weight='bold')
            s.text(0.01*m, 1.08*m, 'Experiment', fontsize=label_fontsize, weight='bold')

            ellipse = matplotlib.patches.Ellipse((x, y), rx, ry, color='b', fill=False)
            s.add_patch(ellipse)
            ellipse = matplotlib.patches.Ellipse((y, x), ry, rx, color='b', fill=False)
            s.add_patch(ellipse)


    for n, ax in enumerate([ax1, ax3, ax5, ax7]):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    plt.tight_layout()

    plt.savefig('/home/erschultz/TICG-chromatin/figures/hic_metrics.png')
    plt.close()

def scc_across_cell_lines():
    dataset = 'dataset_12_06_23'
    samples_imr90, _ = get_samples(dataset, filter_cell_lines=['imr90'])
    samples_gm12878, _ = get_samples(dataset, filter_cell_lines=['gm12878'])
    # assert len(samples_imr90) == len(samples_gm12878), f'{len(samples_imr90)} != {len(samples_gm12878)}'
    print(samples_imr90)
    print(samples_gm12878)
    i=min(samples_imr90)
    j=min(samples_gm12878)
    matches = 0
    scc = SCC(h=5, K=100)
    scc_list = []
    while i < max(samples_imr90) and j < max(samples_gm12878):
        d1 = osp.join('/home/erschultz', dataset, f'samples/sample{i}')
        d2 = osp.join('/home/erschultz', dataset, f'samples/sample{j}')
        result1 = load_import_log(d1)
        result2 = load_import_log(d2)
        str1 = f'chr{result1["chrom"]}:{result1["start"]}-{result1["end"]}'
        str2 = f'chr{result2["chrom"]}:{result2["start"]}-{result2["end"]}'

        if result1['chrom'] < result2['chrom']:
            i += 1
        elif result1['chrom'] > result2['chrom']:
            j += 1
        elif result1['start'] < result2['start']:
            i += 1
        elif result1['start'] > result2['start']:
            j += 1
        else:
            i += 1
            j += 1
            y1 = np.load(osp.join(d1, 'y.npy'))
            y2 = np.load(osp.join(d2, 'y.npy'))
            scc_val = scc.scc(y1, y2, var_stabilized=True)
            scc_list.append(scc_val)
            print(scc_val)
            matches += 1
    print(matches)
    print(np.mean(scc_list), ss.sem(scc_list, nan_policy = 'omit'))

def scc_h():
    dataset = 'dataset_12_06_23'
    samples, _ = get_samples(dataset, filter_cell_lines=['imr90'], test=True)
    me_root = 'optimize_grid_b_200_v_8_spheroid_1.5-max_ent10'
    gnn_root = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN631'

    scc_gnn_dict = defaultdict(list) # h : list of scc
    scc_me_dict = defaultdict(list)
    h_max = 10

    for s in samples:
        print(s)
        s_dir = osp.join('/home/erschultz', dataset, f'samples/sample{s}')
        y = np.load(osp.join(s_dir, 'y.npy'))

        me_dir = osp.join(s_dir, me_root)
        final = get_final_max_ent_folder(me_dir)
        y_me = np.load(osp.join(final, 'y.npy'))

        y_gnn = np.load(osp.join(s_dir, gnn_root, 'y.npy'))

        for h in range(h_max):
            scc = SCC(h=h, K=100)

            corr_me = scc.scc(y, y_me, var_stabilized = True)
            scc_me_dict[h].append(corr_me)

            corr_gnn = scc.scc(y, y_gnn, var_stabilized = True)
            scc_gnn_dict[h].append(corr_gnn)

    X = np.arange(0, h_max)
    mean_gnn = np.empty(len(X))
    mean_me = np.empty(len(X))
    std_gnn = np.empty(len(X))
    std_me = np.empty(len(X))
    for i, h in enumerate(X):
        print(scc_gnn_dict[h])
        mean_gnn[i] = np.mean(scc_gnn_dict[h])
        mean_me[i] = np.mean(scc_me_dict[h])
        std_gnn[i] = np.std(scc_gnn_dict[h])
        std_me[i] = np.std(scc_me_dict[h])

    # plt.plot(X, mean_gnn, label='GNN', c='blue')
    # plt.fill_between(X, mean_gnn - std_gnn, mean_gnn + std_gnn, color='blue', alpha=0.5)
    # plt.plot(X, mean_me, label='Maximum Entropy', c='red')
    # plt.fill_between(X, mean_me - std_me, mean_me + std_me, color='red', alpha=0.5)
    plt.errorbar(X, mean_gnn, yerr = std_gnn, color='red', label='GNN')
    plt.errorbar(X, mean_me, yerr=std_me, color='blue', label='Maximum Entropy')
    plt.legend(loc='upper left')
    plt.ylim(None, 0.95)
    plt.ylabel('SCC', fontsize=16)
    plt.xlabel(r'Filter Span Size, $h$', fontsize=16)
    plt.tight_layout()
    plt.savefig('/home/erschultz/TICG-chromatin/figures/scc_h.png')
    plt.close()





if __name__ == '__main__':
    metric_figure()
    # scc_across_cell_lines()
    # scc_h()
