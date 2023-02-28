import math
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from modify_maxent import get_samples, plaid_dist, simple_histogram
from molar_contact_ratio import molar_contact_ratio
from scipy.stats import norm, skewnorm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.similarity_measures import SCC
from sequences_to_contact_maps.scripts.utils import (DiagonalPreprocessing,
                                                     calc_dist_strat_corr)


def meanDist_comparison():
    datasets = ['dataset_01_26_23', 'dataset_02_16_23']
    # datasets = ['dataset_01_26_23', 'dataset_02_04_23', 'dataset_02_21_23']
    # datasets = ['dataset_02_04_23', 'dataset_02_20_23']
    data_dir = osp.join('/home/erschultz', datasets[0])

    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(len(datasets)) % cmap.N
    colors = cmap(ind)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.get_yaxis().set_visible(False)

    for i, dataset in enumerate(datasets):
        meanDist_list = molar_contact_ratio(dataset, None, False)
        print(f'Retrieved meanDist_list for dataset {dataset}')
        for meanDist in meanDist_list:
            ax.plot(meanDist, c = colors[i], alpha=0.6)
        ax2.plot(np.NaN, np.NaN, label = dataset, c = colors[i])

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)

    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(osp.join(data_dir, 'meanDist_comparison.png'))
    plt.close()

def p_s_comparison(dataset, ID):
    samples, experimental = get_samples(dataset)
    data_dir = osp.join('/home/erschultz', dataset)
    samples = np.array(samples)[:10] # cap at 10

    ncols = 5
    if len(samples) <= 8:
        ncols = 4
    nrows = 2
    for log in [True, False]:
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        fig.set_figheight(6*2)
        fig.set_figwidth(6*3)
        row = 0; col=0
        meanDist_list = molar_contact_ratio(dataset, None, False)
        for sample, meanDist_exp in zip(samples, meanDist_list):
            y = np.load(osp.join(data_dir, f'samples/sample{sample}' ,f'GNN-{ID}-E/k0/replicate1/y.npy'))
            y = y.astype(float)
            y /= np.mean(np.diagonal(y))
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
            rmse = mean_squared_error(meanDist, meanDist_exp, squared = False)

            ax = axes[row][col]
            ax.set_title(f'sample{sample}\nRMSE={np.round(rmse, 5)}', fontsize=16)
            if experimental:
                ax.plot(meanDist_exp, label = 'Experiment', color = 'k')
            else:
                ax.plot(meanDist_exp, label = 'Simulation', color = 'k')
            ax.plot(meanDist, label = 'GNN', color = 'blue')
            ax.legend(loc='upper left')
            ax.set_yscale('log')
            if log:
                ax.set_xscale('log')
                ax.set_xlim(1, 550)

            col += 1
            if col == ncols:
                col = 0
                row += 1

        fig.supylabel('Contact Probability', fontsize = 16)
        fig.supxlabel('Polymer Distance (beads)', fontsize = 16)
        fig.suptitle(f'{dataset}\nGNN={ID}', fontsize=16)
        plt.tight_layout()
        if log:
                plt.savefig(osp.join(data_dir, 'p_s_comparison_log.png'))
        else:
            plt.savefig(osp.join(data_dir, 'p_s_comparison.png'))
        plt.close()

def scc_comparison(dataset, ID, k=8):
    samples, experimental = get_samples(dataset)
    data_dir = osp.join('/home/erschultz', dataset)
    samples = np.array(samples)[:10] # cap at 10
    scc = SCC()

    ncols = 5
    if len(samples) <= 8:
        ncols = 4
    nrows = 2
    for log in [True, False]:
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        fig.set_figheight(6*2)
        fig.set_figwidth(6*3)
        row = 0; col=0
        for sample in samples:
            y = np.load(osp.join(data_dir, f'samples/sample{sample}', 'y.npy'))
            y = y.astype(float)

            yhat = np.load(osp.join(data_dir, f'samples/sample{sample}', f'GNN-{ID}-E/k0/replicate1/y.npy'))
            yhat = yhat.astype(float)

            yhat_pca = np.load(osp.join(data_dir, f'samples/sample{sample}', f'PCA-normalize-E/k{k}/replicate1/y.npy'))

            corr_scc_var_pca = scc.scc(y, yhat_pca, var_stabilized = True)
            _, corr_arr_pca = calc_dist_strat_corr(y, yhat_pca, mode = 'pearson',
                                                    return_arr = True)


            corr_scc_var = scc.scc(y, yhat, var_stabilized = True)
            _, corr_arr = calc_dist_strat_corr(y, yhat, mode = 'pearson',
                                                    return_arr = True)

            ax = axes[row][col]
            ax.set_title(f'sample{sample}\nGNN SCC: {np.round(corr_scc_var, 3)}\nPCA SCC: {np.round(corr_scc_var_pca, 3)}', fontsize=16)
            ax.plot(corr_arr, color = 'b', label = f'GNN {ID}')
            ax.plot(corr_arr_pca, color = 'r', label = f'PCA(k={k})')
            if log:
                ax.set_xscale('log')
                ax.set_xlim(1, 550)
            else:
                ax.set_ylim(-0.5, 1)


            col += 1
            if col == ncols:
                col = 0
                row += 1

        axes[0,0].legend(loc = 'lower left')
        fig.supxlabel('Polymer Distance (beads)', fontsize = 16)
        fig.supylabel('Pearson Correlation Coefficient', fontsize = 16)
        fig.suptitle(f'{dataset}', fontsize=16)
        plt.tight_layout()
        if log:
                plt.savefig(osp.join(data_dir, 'distance_pearson_log.png'))
        else:
            plt.savefig(osp.join(data_dir, 'distance_pearson.png'))
        plt.close()


def l_ij_comparison(dataset):
    data_dir = osp.join('/home/erschultz', dataset)

    L_list = []
    chi_list = []
    label_list = []
    L_max_ent, chi_max_ent = plaid_dist(dataset, 4, False)
    L_list.append(L_max_ent)
    # chi_list.append(chi_max_ent)
    label_list.append('Max Ent')

    s_sim, chi_sim = plaid_dist('dataset_02_22_23', None, False)
    L_list.append(s_sim)
    # chi_list.append(chi_sim)
    label_list.append(r'Synthetic $\tilde{\chi}$')

    # plot plaid L_ij parameters
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(len(L_list)) % cmap.N
    colors = cmap(ind)
    dist = norm
    bin_width = 1
    for i, (arr, label) in enumerate(zip(L_list, label_list)):
        arr = np.array(arr).reshape(-1)
        print(arr)
        print(np.min(arr), np.max(arr))
        _, bins, _ = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins=range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width),
                                    alpha = 0.5, label = label, color = colors[i])
        params = dist.fit(arr)
        y = dist.pdf(bins, *params) * bin_width
        plt.plot(bins, y, ls = '--', color = colors[i])

    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(r'$L_{ij}$', fontsize=16)
    plt.xlim(-20, 20)
    plt.savefig(osp.join(data_dir, 'L_dist_comparison.png'))
    plt.close()

    # plot plaid chi parameters
    # bin_width = 1
    # for i, (arr, label) in enumerate(zip(chi_list, l_list)):
    #     arr = np.array(arr).reshape(-1)
    #     print(np.min(arr), np.max(arr))
    #     n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
    #                                 bins=range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width),
    #                                 alpha = 0.5, label = label, color = colors[i])
    # plt.legend()
    # plt.ylabel('probability', fontsize=16)
    # plt.xlabel(r'$\chi_{ij}$', fontsize=16)
    # plt.xlim(-20, 20)
    # plt.savefig(osp.join(data_dir, 'chi_dist_comparison.png'))
    # plt.close()



if __name__ == '__main__':
    # main()
    meanDist_comparison()
    # l_ij_comparison('dataset_02_04_23')
    # l_dist_comparison()
    # p_s_comparison('dataset_02_04_23', 380)
    # scc_comparison('dataset_02_04_23', 380)
