import math
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from modify_maxent import get_samples, plaid_dist, simple_histogram
from molar_contact_ratio import molar_contact_ratio
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import RED_BLUE_CMAP, RED_CMAP
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import make_composite
from scipy.stats import norm, skewnorm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_Y)
from sequences_to_contact_maps.scripts.utils import (calc_dist_strat_corr,
                                                     triu_to_full)


def meanDist_comparison():
    # datasets = ['dataset_01_26_23', 'dataset_02_16_23']
    # datasets = ['dataset_01_26_23', 'dataset_02_04_23', 'dataset_02_21_23']
    datasets = ['dataset_02_04_23', 'dataset_08_17_23']
    labels = ['Experiment', 'Synthetic']
    data_dir = osp.join('/home/erschultz', datasets[0])

    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(len(datasets)) % cmap.N
    colors = cmap(ind)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.get_yaxis().set_visible(False)

    for i, (dataset, label) in enumerate(zip(datasets, labels)):
        meanDist_list = molar_contact_ratio(dataset, None, False)
        print(f'Retrieved meanDist_list for dataset {dataset}')
        arr = []
        for meanDist in meanDist_list:
            print(meanDist[:10])
            arr.append(meanDist[10])
            ax.plot(meanDist, c = colors[i], alpha=0.6)
        mean = np.mean(arr); std = np.std(arr)
        print(mean + std)
        ax2.plot(np.NaN, np.NaN, label = label, c = colors[i])

    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.axvline(10, c='k')
    # ax.axhline(0.06, c='k')
    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)

    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(osp.join(data_dir, 'meanDist_comparison.png'))
    plt.close()

def p_s_comparison(dataset, GNN_ID, b, phi, k=8, max_ent=False):
    samples, experimental = get_samples(dataset, True)
    data_dir = osp.join('/home/erschultz', dataset)
    samples = np.array(samples)[:10] # cap at 10

    if GNN_ID is None:
        plot_GNN = False
    else:
        plot_GNN = True

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
            print('s', sample)
            s_dir = osp.join(data_dir, f'samples/sample{sample}')
            y_exp = np.load(osp.join(s_dir, 'y.npy'))
            y_exp = y_exp.astype(float)
            y_exp /= np.mean(np.diagonal(y_exp))
            meanDist_exp = DiagonalPreprocessing.genomic_distance_statistics(y_exp)

            if plot_GNN:
                y = np.load(osp.join(s_dir,
                                    f'optimize_grid_b_{b}_phi_{phi}-GNN{GNN_ID}',
                                    'y.npy'))
                y = y.astype(float)
                y /= np.mean(np.diagonal(y))
                meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
                rmse = mean_squared_error(meanDist, meanDist_exp, squared = False)

            max_ent_dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}-max_ent{k}')
            if osp.exists(max_ent_dir):
                y_pca = np.load(osp.join(max_ent_dir, 'iteration30/y.npy'))
                y_pca = y_pca.astype(float)
                y_pca /= np.mean(np.diagonal(y_pca))
                meanDist_pca = DiagonalPreprocessing.genomic_distance_statistics(y_pca)
                plot_PCA = True
            else:
                plot_PCA = False

            ax = axes[row][col]
            if plot_GNN:
                ax.set_title(f'sample{sample}\nGNN RMSE={np.round(rmse, 5)}',
                            fontsize=16)
            else:
                ax.set_title(f'sample{sample}', fontsize=16)

            if experimental:
                ax.plot(meanDist_exp, label = 'Experiment', color = 'k')
            else:
                ax.plot(meanDist_exp, label = 'Simulation', color = 'k')
            if plot_GNN:
                ax.plot(meanDist, label = 'GNN', color = 'blue')
            if plot_PCA:
                ax.plot(meanDist_pca, label = 'PCA', color = 'red')
            # ax.set_ylim(np.percentile(meanDist_exp, 1), None)
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
        if max_ent:
            fig.suptitle(f'{dataset}_max_ent\nGNN={GNN_ID}\nk={k}', fontsize=16)
        else:
            fig.suptitle(f'{dataset}\nGNN={GNN_ID}\nk={k}', fontsize=16)
        plt.tight_layout()
        if log:
            plt.savefig(osp.join(data_dir, 'p_s_comparison_log.png'))
        else:
            plt.savefig(osp.join(data_dir, 'p_s_comparison.png'))
        plt.close()

def scc_comparison(dataset, ID=None, k=8, max_ent=False):
    samples, experimental = get_samples(dataset)
    data_dir = osp.join('/home/erschultz', dataset)
    samples = np.array(samples)[:10] # cap at 10
    scc = SCC()

    if ID is None:
        plot_GNN = False
    else:
        plot_GNN = True

    ncols = 5
    if len(samples) <= 8:
        ncols = 4
    nrows = 2
    for log in [True, False]:
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        fig.set_figheight(6*2)
        fig.set_figwidth(6*3)
        row = 0; col=0
        scc_GNN = []
        scc_PCA = []
        for sample in samples:
            s_dir = osp.join(data_dir, f'samples/sample{sample}')
            if max_ent:
                dir = osp.join(s_dir, f'PCA-normalize-S/k{k}',
                                f'replicate1/samples/sample{sample}_copy')
                if not osp.exists(dir):
                    dir = osp.join(s_dir, f'PCA-normalize-E/k{k}',
                                f'/replicate1/samples/sample{sample}_copy')
            else:
                dir = s_dir
            y = np.load(osp.join(dir, 'y.npy'))
            y = y.astype(float)

            yhat = np.load(osp.join(dir, f'GNN-{ID}-S/k0/replicate1/y.npy'))
            yhat = yhat.astype(float)
            corr_scc_var = scc.scc(y, yhat, var_stabilized = True)
            _, corr_arr = calc_dist_strat_corr(y, yhat, mode = 'pearson',
                                                    return_arr = True)
            scc_GNN.append(corr_scc_var)

            y_file = osp.join(dir ,f'PCA-normalize-S/k{k}/replicate1/y.npy')
            y_file2 = osp.join(dir ,f'PCA-normalize-E/k{k}/replicate1/y.npy')
            if osp.exists(y_file):
                yhat_pca = np.load(y_file)
                corr_scc_var_pca = scc.scc(y, yhat_pca, var_stabilized = True)
                _, corr_arr_pca = calc_dist_strat_corr(y, yhat_pca, mode = 'pearson',
                                                        return_arr = True)
                plot_PCA = True
            elif osp.exists(y_file2):
                yhat_pca = np.load(y_file2)
                corr_scc_var_pca = scc.scc(y, yhat_pca, var_stabilized = True)
                _, corr_arr_pca = calc_dist_strat_corr(y, yhat_pca, mode = 'pearson',
                                                        return_arr = True)
                plot_PCA = True

            else:
                plot_PCA = False

            if plot_PCA:
                scc_PCA.append(corr_scc_var_pca)

            ax = axes[row][col]
            if plot_PCA and plot_GNN:
                title = f'sample{sample}\n'
                title += f'GNN SCC: {np.round(corr_scc_var, 3)}\n'
                title += f'PCA SCC: {np.round(corr_scc_var_pca, 3)}'
                ax.set_title(title, fontsize=16)
            elif plot_GNN:
                ax.set_title(f'sample{sample}\nGNN SCC: {np.round(corr_scc_var, 3)}',
                            fontsize=16)
            else:
                ax.set_title(f'sample{sample}', fontsize=16)
            if plot_GNN:
                ax.plot(corr_arr, color = 'b', label = f'GNN {ID}')
            if plot_PCA:
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
        if max_ent:
            fig.suptitle(f'{dataset}_max_ent\nGNN={ID}\nk={k}', fontsize=16)
        else:
            fig.suptitle(f'{dataset}\nGNN={ID}\nk={k}', fontsize=16)
        plt.tight_layout()
        if log:
                plt.savefig(osp.join(data_dir, 'distance_pearson_log.png'))
        else:
            plt.savefig(osp.join(data_dir, 'distance_pearson.png'))
        plt.close()

    print(f'GNN={np.mean(scc_GNN)}, PCA={np.mean(scc_PCA)}')


def l_ij_comparison(dataset, dataset_exp, k=8):
    data_dir = osp.join('/home/erschultz', dataset)

    L_list = []
    D_list = []
    S_list = []
    chi_list = []
    label_list = []
    L_max_ent, S_max_ent, D_max_ent, chi_max_ent = plaid_dist(dataset_exp, k, False)
    L_list.append(L_max_ent)
    D_list.append(D_max_ent)
    S_list.append(S_max_ent)
    # chi_list.append(chi_max_ent)
    label_list.append('Max Ent')

    L_sim, S_sim, D_sim, chi_sim = plaid_dist(dataset, None, False)
    L_list.append(L_sim)
    D_list.append(D_sim)
    S_list.append(S_sim)

    # chi_list.append(chi_sim)
    label_list.append(r'Synthetic')

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
        bin_positions = range(math.floor(min(arr)),
                                math.ceil(max(arr)) + bin_width,
                                bin_width)
        _, bins, _ = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins=bin_positions,
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

    # plot D_ij parameters
    print('\nD')
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(len(D_list)) % cmap.N
    colors = cmap(ind)
    dist = norm
    bin_width = 1
    for i, (arr, label) in enumerate(zip(D_list, label_list)):
        arr = np.array(arr).reshape(-1)
        print(label)
        print(np.min(arr), np.max(arr))
        bin_positions = range(math.floor(min(arr)),
                                math.ceil(max(arr)) + bin_width,
                                bin_width)

        _, bins, _ = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins=bin_positions,
                                    alpha = 0.5, label = label, color = colors[i])
        params = dist.fit(arr)
        y = dist.pdf(bins, *params) * bin_width
        plt.plot(bins, y, ls = '--', color = colors[i])

    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(r'$D_{ij}$', fontsize=16)
    plt.xlim(-20, 25)
    plt.savefig(osp.join(data_dir, 'D_dist_comparison.png'))
    plt.close()

    # plot plaid S_ij parameters
    print('\nS')
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(len(L_list)) % cmap.N
    colors = cmap(ind)
    dist = norm
    bin_width = 1
    for i, (arr, label) in enumerate(zip(S_list, label_list)):
        arr = np.array(arr).reshape(-1)
        print(arr)
        print(np.min(arr), np.max(arr))
        bin_positions = range(math.floor(min(arr)),
                                math.ceil(max(arr)) + bin_width,
                                bin_width)
        _, bins, _ = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins=bin_positions,
                                    alpha = 0.5, label = label, color = colors[i])
        params = dist.fit(arr)
        y = dist.pdf(bins, *params) * bin_width
        plt.plot(bins, y, ls = '--', color = colors[i])

    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(r'$S_{ij}$', fontsize=16)
    plt.xlim(-20, 25)
    plt.savefig(osp.join(data_dir, 'S_dist_comparison.png'))
    plt.close()

    # plot frob(L)
    print('\nL frob norm')
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(len(L_list)) % cmap.N
    colors = cmap(ind)
    dist = norm
    bin_width = 100
    for i, (sublist, label) in enumerate(zip(L_list, label_list)):
        arr = []
        for L in sublist:
            L = triu_to_full(L)
            arr.append(np.linalg.norm(L, ord='fro'))

        arr = np.array(arr).reshape(-1)
        print(arr)
        print(np.min(arr), np.max(arr))
        bin_positions = range(math.floor(min(arr)),
                                math.ceil(max(arr)) + bin_width,
                                bin_width)
        _, bins, _ = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                bins=bin_positions,
                                alpha = 0.5, label = label, color = colors[i])
        params = dist.fit(arr)
        y = dist.pdf(bins, *params) * bin_width
        plt.plot(bins, y, ls = '--', color = colors[i])

    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel('Frob(L)', fontsize=16)
    plt.savefig(osp.join(data_dir, 'frob_L_comparison.png'))
    plt.close()

def plot_y_S(dataset, b, phi):
    dir = '/project2/depablo/erschultz/'
    if not osp.exists(dir):
        dir = '/home/erschultz'
    data_dir = osp.join(dir, dataset)

    odir = osp.join(data_dir, f'b_{b}_phi_{phi}_distributions')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)

    samples, _ = get_samples(dataset, train=True)
    N = 10
    samples = np.array(samples)[:N]

    y_list = []
    composite_list = []
    S_list = []
    for i, sample in enumerate(samples):
        sample_dir = osp.join(data_dir, f'samples/sample{sample}')

        y, y_diag = load_Y(sample_dir)
        y /= np.mean(np.diagonal(y))
        y_list.append(y)


        max_ent_dir = osp.join(sample_dir, f'optimize_grid_b_{b}_phi_{phi}-max_ent10')
        final = get_final_max_ent_folder(max_ent_dir)
        S = np.load(osp.join(final, 'S.npy'))
        S_list.append(S)
        y_hat = np.load(osp.join(final, 'y.npy'))

        composite = make_composite(y, y_hat)
        composite_list.append(composite)


    S_dag_list = [np.sign(S) * np.log(np.abs(S)+1) for S in S_list]

    # plot contact maps
    rows=2; cols=5
    fig, ax = plt.subplots(rows, cols+1,
                            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
    fig.set_figheight(6*2)
    fig.set_figwidth(6*3)
    vmin = 0; vmax = np.mean(y_list)
    row = 0; col=0
    for y, sample in zip(y_list, samples):
        if col == 0:
            s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax,
                        cmap = RED_CMAP, ax = ax[row][col], cbar_ax = ax[row][-1])
        else:
            s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax,
                        cmap = RED_CMAP, ax = ax[row][col], cbar = False)
        s.set_title(f'Sample {sample}', fontsize = 16)
        s.set_xticks([])
        s.set_yticks([])

        col += 1
        if col == cols:
            col = 0
            row += 1

    plt.tight_layout()
    plt.savefig(osp.join(data_dir, 'y_list.png'))
    plt.close()

    # # plot composite contact maps
    # rows=2; cols=5
    # fig, ax = plt.subplots(rows, cols+1,
    #                         gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
    # fig.set_figheight(6*2)
    # fig.set_figwidth(6*3)
    # vmin = 0; vmax = np.mean(y_list)
    # row = 0; col=0
    # for y, sample in zip(composite_list, samples):
    #     if col == 0:
    #         s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax,
    #                     cmap = RED_CMAP, ax = ax[row][col], cbar_ax = ax[row][-1])
    #     else:
    #         s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax,
    #                     cmap = RED_CMAP, ax = ax[row][col], cbar = False)
    #
    #     s.axline((0,0), slope=1, color = 'k', lw=1)
    #     m, _ = y.shape
    #     s.text(0.99*m, 0.01*m, 'Max Ent', fontsize=16, ha='right', va='top')
    #     s.text(0.01*m, 0.99*m, 'Experiment', fontsize=16)
    #     s.set_title(f'Sample {sample}', fontsize = 16)
    #     s.set_xticks([])
    #     s.set_yticks([])
    #
    #     col += 1
    #     if col == cols:
    #         col = 0
    #         row += 1
    #
    # plt.tight_layout()
    # plt.savefig(osp.join(odir, 'y_composite_list.png'))
    # plt.close()
    #
    # # plot S orderd by rab
    # fig, ax = plt.subplots(rows, cols+1,
    #                         gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
    # fig.set_figheight(6*2)
    # fig.set_figwidth(6*3)
    # vmin = np.nanpercentile(S_list, 1)
    # vmax = np.nanpercentile(S_list, 99)
    # vmax = max(vmax, vmin * -1)
    # vmin = vmax * -1
    # row = 0; col=0
    # for S, sample in zip(S_list, samples):
    #     if col == 0:
    #         s = sns.heatmap(S, linewidth = 0, vmin = vmin, vmax = vmax,
    #                         cmap = RED_BLUE_CMAP,
    #             ax = ax[row][col], cbar_ax = ax[row][-1])
    #     else:
    #         s = sns.heatmap(S, linewidth = 0, vmin = vmin, vmax = vmax,
    #                         cmap = RED_BLUE_CMAP,
    #             ax = ax[row][col], cbar = False)
    #     s.set_title(f'Sample {sample}', fontsize = 16)
    #     s.set_xticks([])
    #     s.set_yticks([])
    #
    #     col += 1
    #     if col == cols:
    #         col = 0
    #         row += 1
    #
    # plt.tight_layout()
    # plt.savefig(osp.join(odir, 'S_list.png'))
    # plt.close()
    #
    # # plot S_dag orderd by rab
    # S_dag_arr = np.array([np.sign(S) * np.log(np.abs(S)+1) for S in S_list])
    # fig, ax = plt.subplots(rows, cols+1,
    #                         gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
    # fig.set_figheight(6*2)
    # fig.set_figwidth(6*3)
    # vmin = np.nanpercentile(S_dag_arr, 1)
    # vmax = np.nanpercentile(S_dag_arr, 99)
    # vmax = max(vmax, vmin * -1)
    # vmin = vmax * -1
    # row = 0; col=0
    # for S_dag, sample in zip(S_dag_list, samples):
    #     if col == 0:
    #         s = sns.heatmap(S_dag, linewidth = 0, vmin = vmin, vmax = vmax,
    #                         cmap = RED_BLUE_CMAP,
    #             ax = ax[row][col], cbar_ax = ax[row][-1])
    #     else:
    #         s = sns.heatmap(S_dag, linewidth = 0, vmin = vmin, vmax = vmax,
    #                         cmap = RED_BLUE_CMAP,
    #             ax = ax[row][col], cbar = False)
    #     s.set_title(f'Sample {sample}', fontsize = 16)
    #     s.set_xticks([])
    #     s.set_yticks([])
    #
    #     col += 1
    #     if col == cols:
    #         col = 0
    #         row += 1
    #
    # plt.tight_layout()
    # plt.savefig(osp.join(odir, 'S_dag_list.png'))
    # plt.close()

    # plot S_center orderd by rab
    S_center_arr = np.array([S - np.mean(S) for S in S_list])
    fig, ax = plt.subplots(rows, cols+1,
                            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
    fig.set_figheight(6*2)
    fig.set_figwidth(6*3)
    vmin = np.nanpercentile(S_center_arr, 1)
    vmax = np.nanpercentile(S_center_arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    row = 0; col=0
    for S, sample in zip(S_center_arr, samples):
        if col == 0:
            s = sns.heatmap(S, linewidth = 0, vmin = vmin, vmax = vmax,
                            cmap = RED_BLUE_CMAP,
                ax = ax[row][col], cbar_ax = ax[row][-1])
        else:
            s = sns.heatmap(S, linewidth = 0, vmin = vmin, vmax = vmax,
                            cmap = RED_BLUE_CMAP,
                ax = ax[row][col], cbar = False)
        s.set_title(f'Sample {sample}', fontsize = 16)
        s.set_xticks([])
        s.set_yticks([])

        col += 1
        if col == cols:
            col = 0
            row += 1

    plt.tight_layout()
    plt.savefig(osp.join(odir, 'S_center_list.png'))
    plt.close()

    # plot S_center_dag ordered by rab
    S_center_dag_arr = np.array([np.sign(S) * np.log(np.abs(S)+1) for S in S_center_arr])
    fig, ax = plt.subplots(rows, cols+1,
                            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
    fig.set_figheight(6*2)
    fig.set_figwidth(6*3)
    vmin = np.nanpercentile(S_center_dag_arr, 1)
    vmax = np.nanpercentile(S_center_dag_arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    row = 0; col=0
    for S_dag, sample in zip(S_center_dag_arr, samples):
        if col == 0:
            s = sns.heatmap(S_dag, linewidth = 0, vmin = vmin, vmax = vmax,
                            cmap = RED_BLUE_CMAP,
                ax = ax[row][col], cbar_ax = ax[row][-1])
        else:
            s = sns.heatmap(S_dag, linewidth = 0, vmin = vmin, vmax = vmax,
                            cmap = RED_BLUE_CMAP,
                ax = ax[row][col], cbar = False)
        s.set_title(f'Sample {sample}', fontsize = 16)
        s.set_xticks([])
        s.set_yticks([])

        col += 1
        if col == cols:
            col = 0
            row += 1

    plt.tight_layout()
    plt.savefig(osp.join(odir, 'S_center_dag_list.png'))
    plt.close()




if __name__ == '__main__':
    # main()
    # plot_y_S('dataset_02_04_23', 140, 0.03)
    plot_y_S('dataset_02_04_23', 261, 0.01)

    # meanDist_comparison()
    # l_ij_comparison('dataset_04_28_23', 'dataset_02_04_23', 10)
    # p_s_comparison('dataset_02_04_23', None, 261, 0.01, 10)
    # scc_comparison('dataset_02_04_23', 392, 8, True)
