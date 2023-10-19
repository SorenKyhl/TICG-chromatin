import json
import multiprocessing as mp
import os
import os.path as osp
import string
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as ss
import seaborn as sns
from pylib.utils.plotting_utils import RED_CMAP, plot_matrix
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import triu_to_full

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_import_log)

sys.path.append('/home/erschultz/TICG-chromatin')
import GNN
import max_ent

from scripts.data_generation.modify_maxent import get_samples

EXP_DATASET='dataset_02_04_23'

def make_samples():
    exponents = np.arange(4, 9)
    dir = '/home/erschultz/downsampling_analysis'
    for exponent in exponents:
        e_dir = f'{dir}/samples_exp{exponent}'
        if not osp.exists(e_dir):
            os.mkdir(e_dir, mode=0o755)

    tot_count_list = []
    samples, _ = get_samples(EXP_DATASET, test=True)
    for s_exp in samples[:10]:
        print(s_exp)
        exp_dir = f'/home/erschultz/{EXP_DATASET}/samples/sample{s_exp}'
        y = np.triu(np.load(osp.join(exp_dir, 'y.npy')))
        m = len(y)
        y_flat = y[np.triu_indices(m)]
        p_flat = y_flat / np.sum(y_flat)
        pos = np.arange(0, len(p_flat))

        exp_dir = f'/home/erschultz/{EXP_DATASET}/samples_10k/sample{s_exp-200}'

        y = np.triu(np.load(osp.join(exp_dir, 'y.npy')))
        tot_count = np.sum(y)
        print(f'Total Read Depth: {tot_count}')
        tot_count_list.append(tot_count)

        for exponent in exponents:
            e_dir = f'{dir}/samples_exp{exponent}'
            print(exponent)
            odir = osp.join(e_dir, f'sample{s_exp}')
            if osp.exists(odir):
                continue
            os.mkdir(odir, mode = 0o755)
            count = 10**exponent
            choices = np.random.choice(pos, size = count, p = p_flat)
            y_i_flat = np.zeros_like(p_flat)
            for j in choices:
                # this is really slow
                y_i_flat[j] += 1
            print(np.sum(y_i_flat))
            y_i = triu_to_full(y_i_flat)
            np.save(osp.join(odir, 'y.npy'), y_i)
            plot_matrix(y_i, osp.join(odir, 'y.png'), vmax = 'mean')
    tot_count_mean = np.mean(tot_count_list)
    print(f'Mean Total Read Depth: {tot_count_mean}')

def fit_gnn(GNN_id):
    dataset='downsampling_analysis'
    samples, _ = get_samples(EXP_DATASET, test=True)
    N = 10
    samples = samples[:N]

    GNN_IDs = [GNN_id]
    for downsampling in [4, 5, 6, 7, 8]:
        mapping = []

        for GNN_ID in GNN_IDs:
            for i in samples:
                mapping.append((dataset, i, GNN_ID, f'samples_exp{downsampling}', 140, 0.03))

        print(len(mapping))
        print(mapping)

        # this must be nested because of how GNN uses scratch
        with mp.Pool(N) as p:
            p.starmap(GNN.fit, mapping)

def fit_max_ent():
    dataset='downsampling_analysis'
    samples, _ = get_samples(EXP_DATASET, test=True)
    N = 10
    samples = samples[:N]

    mapping = []
    for downsampling in [4, 5, 6, 7, 8]:
        for i in samples:
            mapping.append((dataset, i, f'samples_exp{downsampling}',
                            180, None, 8, None, 1.5))

    print(len(mapping))
    print(mapping)

    with mp.Pool(15) as p:
        p.starmap(max_ent.fit, mapping)


def figure(GNN_ID):
    label_fontsize=24
    tick_fontsize=22
    letter_fontsize=26

    samples, _ = get_samples(EXP_DATASET, test=True)
    N = 10
    samples = samples[:N]
    exponents = np.arange(4, 9)
    read_counts = 10**exponents
    dir = '/home/erschultz/downsampling_analysis'
    m=512
    gnn_scc_dense = defaultdict(lambda: np.zeros(N))
                    # exponent : list of sccs
    max_ent_scc_dense = defaultdict(lambda: np.zeros(N))
    experiment_scc_dense = defaultdict(lambda: np.zeros(N))
    gnn_scc_sparse = defaultdict(lambda: np.zeros(N))
    max_ent_scc_sparse = defaultdict(lambda: np.zeros(N))
    composite_dict = defaultdict(lambda: np.zeros((N,m,m)))
    max_ent_composite_dict = defaultdict(lambda: np.zeros((N,m,m)))

    y_list = [] # ground truth y from highly sampled simulation
    for s in samples:
        y = np.load(f'/home/erschultz/{EXP_DATASET}/samples/sample{s}/y.npy')
        y /= np.mean(y.diagonal())
        y_list.append(y)

    # collect data
    scc = SCC(h=1)
    for exp in exponents:
        for i, s in enumerate(samples):
            s_dir = osp.join(dir, f'samples_exp{exp}/sample{s}')
            y_sparse = np.load(osp.join(s_dir, 'y.npy'))
            y_sparse /= np.mean(y_sparse.diagonal())
            y_dense = y_list[i]
            m = len(y_sparse)
            indu = np.triu_indices(m)
            indl = np.tril_indices(m)


            # experiment
            corr_scc_var = scc.scc(y_dense, y_sparse)
            experiment_scc_dense[exp][i] = corr_scc_var

            # gnn
            gnn_dir = osp.join(s_dir, f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}')
            yhat = np.load(osp.join(gnn_dir, 'y.npy'))
            gnn_scc_dense[exp][i] = scc.scc(y_dense, yhat)
            gnn_scc_sparse[exp][i] = scc.scc(y_sparse, yhat)

            # make composite contact map
            composite = np.zeros((m, m))
            composite[indu] = yhat[indu]
            composite[indl] = y_sparse[indl]
            composite_dict[exp][i] = composite

            # max ent
            max_ent_dir = osp.join(s_dir, 'optimize_grid_b_140_phi_0.03-max_ent10')
            final = get_final_max_ent_folder(max_ent_dir)
            yhat = np.load(osp.join(final, 'y.npy'))
            max_ent_scc_dense[exp][i] = scc.scc(y_dense, yhat)
            max_ent_scc_sparse[exp][i] = scc.scc(y_sparse, yhat)

            # make max ent composite contact map
            composite = np.zeros((m, m))
            composite[indu] = yhat[indu]
            composite[indl] = y_sparse[indl]
            max_ent_composite_dict[exp][i] = composite

    # compute statistics
    gnn_scc_stats = defaultdict(
                            lambda: defaultdict(
                            lambda: np.zeros_like(exponents, dtype=float)))
                            # {sparse, dense} : {mean, std} : list of values
    max_ent_scc_stats = defaultdict(
                            lambda: defaultdict(
                            lambda: np.zeros_like(exponents, dtype=float)))
    experiment_scc_stats = defaultdict(
                            lambda: defaultdict(
                            lambda: np.zeros_like(exponents, dtype=float)))
    for i, exp in enumerate(exponents):
        # sparse
        gnn_scc_stats['sparse']['mean'][i] = np.mean(gnn_scc_sparse[exp])
        max_ent_scc_stats['sparse']['mean'][i] = np.mean(max_ent_scc_sparse[exp])
        gnn_scc_stats['sparse']['std'][i] = np.std(gnn_scc_sparse[exp], ddof=1)
        max_ent_scc_stats['sparse']['std'][i] = np.std(max_ent_scc_sparse[exp], ddof=1)
        gnn_scc_stats['sparse']['sem'][i] = ss.sem(gnn_scc_sparse[exp])
        max_ent_scc_stats['sparse']['sem'][i] = ss.sem(max_ent_scc_sparse[exp])

        # dense
        gnn_scc_stats['dense']['mean'][i] = np.mean(gnn_scc_dense[exp])
        max_ent_scc_stats['dense']['mean'][i]= np.mean(max_ent_scc_dense[exp])
        experiment_scc_stats['dense']['mean'][i] = np.mean(experiment_scc_dense[exp])
        gnn_scc_stats['dense']['std'][i] = np.std(gnn_scc_dense[exp], ddof=1)
        max_ent_scc_stats['dense']['std'][i]= np.std(max_ent_scc_dense[exp], ddof=1)
        gnn_scc_stats['dense']['sem'][i] = ss.sem(gnn_scc_dense[exp])
        max_ent_scc_stats['dense']['sem'][i] = ss.sem(max_ent_scc_dense[exp])
        experiment_scc_stats['dense']['std'][i] = np.std(experiment_scc_dense[exp], ddof=1)
        experiment_scc_stats['dense']['sem'][i] = ss.sem(experiment_scc_dense[exp])

    print(experiment_scc_stats['dense'])
    print(gnn_scc_stats['sparse'])

    ### combined plot ##
    # select which composites to plot
    sample_i = 4
    sample = samples[sample_i]
    exponents_hic = [5, 6, 7]
    result = load_import_log(f'/home/erschultz/{EXP_DATASET}/samples/sample{sample}')
    start = result['start_mb']
    end = result['end_mb']
    chrom = result['chrom']
    print(f'Chr{chrom}:{start}-{end}Mb')
    resolution = result['resolution']
    m = len(y)
    all_labels = np.linspace(start, end, m)
    all_labels = np.round(all_labels, 1)
    genome_ticks = [0, m//3, 2*m//3, m-1]
    genome_labels = [f'{all_labels[i]}' for i in genome_ticks]

    fig = plt.figure(figsize=(14, 16))
    ax1 = plt.subplot(3, 24, (1, 8))
    ax2 = plt.subplot(3, 24, (9, 16))
    ax3 = plt.subplot(3, 24, (17, 24))
    ax4 = plt.subplot(3, 24, (25, 32))
    ax5 = plt.subplot(3, 24, (33, 40))
    ax6 = plt.subplot(3, 24, (41, 48))
    ax7 = plt.subplot(3, 2, 5) # scc sparse
    ax8 = plt.subplot(3, 2, 6) # scc dense
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    # hic
    composites = np.zeros((len(exponents_hic), m, m))
    for i, exp in enumerate(exponents_hic):
        composites[i] = composite_dict[exp][sample_i]
    vmax = np.mean(composites)
    for i, (exp, composite) in enumerate(zip(exponents_hic, composites)):
        scc = gnn_scc_sparse[exp][sample_i]
        scc = np.round(scc, 3)
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax,
                        cmap = RED_CMAP, ax = axes[i], cbar = False)
        s.set_title(f'Read Depth = $10^{{{exp}}}$\nSCC={scc}',
                    fontsize = label_fontsize)
        # s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
        #             fontsize = tick_fontsize)
        # s.set_yticks(genome_ticks, labels = genome_labels, rotation = 0,
        #             fontsize = tick_fontsize)
        s.set_xticks([])
        s.set_yticks([])
        axes[i].axline((0,0), slope=1, color = 'k', lw=1)
        axes[i].text(0.99*m, 0.01*m, 'GNN', fontsize=letter_fontsize, ha='right',
                    va='top', weight='bold')
        axes[i].text(0.01*m, 0.99*m, 'Reference', fontsize=letter_fontsize,
                    weight='bold')

    composites = np.zeros((len(exponents_hic), m, m))
    for i, exp in enumerate(exponents_hic):
        composites[i] = max_ent_composite_dict[exp][sample_i]
    vmax = np.mean(composites)
    for i, (exp, composite) in enumerate(zip(exponents_hic, composites)):
        ax = axes[i+3]
        scc = max_ent_scc_sparse[exp][sample_i]
        scc = np.round(scc, 3)
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax,
                        cmap = RED_CMAP, ax = ax, cbar = False)
        s.set_title(f'SCC={scc}', fontsize = label_fontsize)
        # s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
        #             fontsize = tick_fontsize)
        # s.set_yticks(genome_ticks, labels = genome_labels, rotation = 0,
        #             fontsize = tick_fontsize)
        s.set_xticks([])
        s.set_yticks([])
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*m, 0.01*m, 'Max Ent', fontsize=letter_fontsize, ha='right',
                va='top', weight='bold')
        ax.text(0.01*m, 0.99*m, 'Reference', fontsize=letter_fontsize, weight='bold')

    # scc
    for ax, mode in zip([ax7, ax8], ['sparse', 'dense']):
        print(mode)
        if mode == 'dense':
            experiment_mean = experiment_scc_stats[mode]['mean']
            experiment_std = experiment_scc_stats[mode]['std']
            ax.errorbar(read_counts, experiment_mean, experiment_std,
                    color='k', label='Subsampled Experiment')

        max_ent_mean = max_ent_scc_stats[mode]['mean']
        print(max_ent_mean)
        max_ent_std = max_ent_scc_stats[mode]['sem']
        ax.errorbar(read_counts, max_ent_mean, max_ent_std,
                        color='blue', label = 'Maximum Entropy')

        gnn_mean = gnn_scc_stats[mode]['mean']
        print(gnn_mean)
        gnn_std = gnn_scc_stats[mode]['sem']
        ax.errorbar(read_counts, gnn_mean, gnn_std,
                        color='red', label='GNN')


        ax.set_xlabel('Read Depth', fontsize=label_fontsize)
        ax.set_xscale('log')
        ax.set_xticks(read_counts)

        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ax7.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax8.set_yticks([])
    ax7.set_ylabel('SCC (subsampled)', fontsize=label_fontsize)
    ax8.set_ylabel('SCC (original)', fontsize=label_fontsize)

    ax8.legend(bbox_to_anchor=(-0.15, -0.25), loc="upper center",
                fontsize = label_fontsize,
                borderaxespad=0, ncol = 3)


    for i, ax in enumerate([ax1, ax4, ax7, ax8]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[i], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')


    # plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.savefig('/home/erschultz/downsampling_analysis/downsampling_figure_exp.png')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/downsampling_figure_exp.png')

    plt.close()




if __name__ == '__main__':
    # make_samples()
    fit_max_ent()
    # fit_gnn(434)
    # figure(434)
