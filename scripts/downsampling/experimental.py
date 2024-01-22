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
from pylib.utils.hic_utils import rescale_p_s_1
from pylib.utils.plotting_utils import RED_CMAP, plot_matrix
from pylib.utils.similarity_measures import SCC, hic_spector
from pylib.utils.utils import make_composite, triu_to_full

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_import_log)

sys.path.append('/home/erschultz/TICG-chromatin')
import scripts.GNN as GNN
import scripts.max_ent as max_ent
from scripts.data_generation.modify_maxent import get_samples

EXP_DATASET='dataset_12_06_23'

def make_samples():
    exponents = np.arange(4, 9)
    dir = '/home/erschultz/downsampling_analysis'
    for exponent in exponents:
        e_dir = f'{dir}/samples_exp{exponent}'
        if not osp.exists(e_dir):
            os.mkdir(e_dir, mode=0o755)

    tot_count_list = []
    samples, _ = get_samples(EXP_DATASET, test=True, filter_cell_lines=['imr90'])
    for s_exp in samples[:10]:
        print(s_exp)
        exp_dir = f'/home/erschultz/{EXP_DATASET}/samples/sample{s_exp}'
        y = np.triu(np.load(osp.join(exp_dir, 'y.npy')))
        m = len(y)
        y_flat = y[np.triu_indices(m)]
        p_flat = y_flat / np.sum(y_flat)
        pos = np.arange(0, len(p_flat))

        exp_dir = f'/home/erschultz/{EXP_DATASET}/samples/sample{s_exp}'
        y = np.triu(np.load(osp.join(exp_dir, 'y.npy')))
        tot_count = np.sum(y)
        print(f'Total Contacts: {tot_count}')
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

            # rescale
            y_i = rescale_p_s_1(y_i, 1e-1)

            np.save(osp.join(odir, 'y.npy'), y_i)
            plot_matrix(y_i, osp.join(odir, 'y.png'), vmax = 'mean')
    tot_count_mean = np.mean(tot_count_list)
    print(f'Mean Total Contacts: {np.round(tot_count_mean, 1)}')

def fit_gnn(GNN_id):
    dataset='downsampling_analysis'
    samples, _ = get_samples(EXP_DATASET, test=True, filter_cell_lines=['imr90'])
    N = 10
    samples = samples[:N]

    GNN_IDs = [GNN_id]
    for downsampling in [4, 5, 6, 7, 8]:
        mapping = []

        for GNN_ID in GNN_IDs:
            for i in samples:
                mapping.append((dataset, i, GNN_ID, f'samples_exp{downsampling}', 200, None, 8, 1.5))

        print(len(mapping))
        print(mapping)

        # this must be nested because of how GNN uses scratch
        with mp.Pool(1) as p:
            p.starmap(GNN.check, mapping)

def fit_max_ent():
    dataset='downsampling_analysis'
    samples, _ = get_samples(EXP_DATASET, test=True, filter_cell_lines=['imr90'])
    N = 10
    samples = samples[:N]

    mapping = []
    for downsampling in [4, 5, 6, 7, 8]:
        for i in samples:
            mapping.append((dataset, i, f'samples_exp{downsampling}',
                            200, None, 8, None, 1.5))

    print(len(mapping))
    print(mapping)

    with mp.Pool(1) as p:
        p.starmap(max_ent.check, mapping)


def figure(GNN_ID):
    label_fontsize=22
    tick_fontsize=18
    letter_fontsize=26

    samples, _ = get_samples(EXP_DATASET, test=True, filter_cell_lines=['imr90'])
    N = 10
    samples = samples[:N]
    exponents = np.arange(4, 9)
    read_counts = 10**exponents
    dir = '/home/erschultz/downsampling_analysis'
    m=512
    gnn_data = defaultdict( # dense / sparse
                lambda: defaultdict( # metric
                lambda: defaultdict( # exponent
                lambda: np.zeros(N)))) # array of values
    max_ent_data =  defaultdict(
                lambda: defaultdict(
                lambda: defaultdict(
                lambda: np.zeros(N))))
    experiment_data =  defaultdict(
                lambda: defaultdict(
                lambda: defaultdict(
                lambda: np.zeros(N))))

    composite_dict = defaultdict(lambda: np.zeros((N,m,m)))
    max_ent_composite_dict = defaultdict(lambda: np.zeros((N,m,m)))

    y_list = [] # ground truth y from highly sampled simulation
    for s in samples:
        y = np.load(f'/home/erschultz/{EXP_DATASET}/samples/sample{s}/y.npy')
        y /= np.mean(y.diagonal())
        y_list.append(y)

    # collect data
    scc = SCC(h=5, K=100)
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
            experiment_data['dense']['scc'][exp][i] = scc.scc(y_dense, y_sparse)
            experiment_data['dense']['spector'][exp][i] = hic_spector(y_dense, y_sparse, 10)

            grid_root = 'optimize_grid_b_200_v_8_spheroid_1.5'

            # gnn
            gnn_dir = osp.join(s_dir, f'{grid_root}-GNN{GNN_ID}')
            y_gnn = np.load(osp.join(gnn_dir, 'y.npy'))
            composite_dict[exp][i] = make_composite(y_sparse, y_gnn)

            # max ent
            max_ent_dir = osp.join(s_dir, f'{grid_root}-max_ent10')
            final = get_final_max_ent_folder(max_ent_dir)
            y_me = np.load(osp.join(final, 'y.npy'))
            max_ent_composite_dict[exp][i] =  make_composite(y_sparse, y_me)

            for mode, y_gt in zip(['sparse', 'dense'], [y_sparse, y_dense]):
                for data_dict, yhat in zip([gnn_data, max_ent_data], [y_gnn, y_me]):
                    data_dict[mode]['scc'][exp][i] = scc.scc(y_gt, yhat)
                    data_dict[mode]['spector'][exp][i] = hic_spector(y_gt, yhat, 10)

    ### combined plot ##
    # select which composites to plot
    sample_i = 4
    sample = samples[sample_i]
    print('sample', sample)
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
    ax7 = plt.subplot(3, 2, 5) # scc dense
    ax8 = plt.subplot(3, 2, 6) # hic spector
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    # hic
    composites = np.zeros((len(exponents_hic), m, m))
    for i, exp in enumerate(exponents_hic):
        composites[i] = composite_dict[exp][sample_i]
    vmax = np.mean(composites)
    for i, (exp, composite) in enumerate(zip(exponents_hic, composites)):
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax,
                        cmap = RED_CMAP, ax = axes[i], cbar = False)
        s.set_title(f'Total Contacts = $10^{{{exp}}}$\n',
                    fontsize = label_fontsize)
        # s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
        #             fontsize = tick_fontsize)
        # s.set_yticks(genome_ticks, labels = genome_labels, rotation = 0,
        #             fontsize = tick_fontsize)
        s.set_xticks([])
        s.set_yticks([])
        axes[i].axline((0,0), slope=1, color = 'k', lw=1)
        axes[i].text(0.99*m, -0.08*m, 'GNN', fontsize=label_fontsize, ha='right',
                    va='top', weight='bold')
        axes[i].text(0.01*m, 1.08*m, 'Reference', fontsize=label_fontsize,
                    weight='bold')

    composites = np.zeros((len(exponents_hic), m, m))
    for i, exp in enumerate(exponents_hic):
        composites[i] = max_ent_composite_dict[exp][sample_i]
    vmax = np.mean(composites)
    for i, (exp, composite) in enumerate(zip(exponents_hic, composites)):
        ax = axes[i+3]
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax,
                        cmap = RED_CMAP, ax = ax, cbar = False)
        #s.set_title(f'SCC={scc}', fontsize = label_fontsize)
        # s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
        #             fontsize = tick_fontsize)
        # s.set_yticks(genome_ticks, labels = genome_labels, rotation = 0,
        #             fontsize = tick_fontsize)
        s.set_xticks([])
        s.set_yticks([])
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*m, -0.08*m, 'Max Ent', fontsize=label_fontsize, ha='right',
                va='top', weight='bold')
        ax.text(0.01*m, 1.08*m, 'Reference', fontsize=label_fontsize, weight='bold')

    # scc
    mode='dense'
    for ax, metric in zip([ax7, ax8], ['scc', 'spector']):
        if False:
            data_mean = np.zeros_like(exponents, dtype=float)
            data_sem = np.zeros_like(exponents, dtype=float)
            for i, exp in enumerate(exponents):
                data_mean[i] = np.mean(experiment_data[mode][metric][exp])
                data_sem[i] = ss.sem(experiment_data[mode][metric][exp])
            ax.errorbar(read_counts, data_mean, data_sem,
                    color='k', label='Subsampled Experiment')

        for label, color, data_dict in zip(['Maximum Entropy','GNN'],
                                            ['blue', 'red'],
                                            [max_ent_data, gnn_data]):
            data_mean = np.zeros_like(exponents, dtype=float)
            data_sem = np.zeros_like(exponents, dtype=float)
            for i, exp in enumerate(exponents):
                data_mean[i] = np.mean(data_dict[mode][metric][exp])
                data_sem[i] = ss.sem(data_dict[mode][metric][exp])
            print(label, data_mean)
            ax.errorbar(read_counts, data_mean, data_sem,
                            color=color, label = label)

        ax.set_xlabel('Total Contacts', fontsize=label_fontsize)
        ax.set_xscale('log')
        ax.set_xticks(read_counts)

        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ax7.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax8.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax7.set_ylabel('SCC', fontsize=label_fontsize)
    ax8.set_ylabel('HiC-Spector', fontsize=label_fontsize)

    ax8.legend(bbox_to_anchor=(-0.15, -0.25), loc="upper center",
                fontsize = label_fontsize,
                borderaxespad=0, ncol = 3)


    for i, ax in enumerate([ax1, ax4, ax7, ax8]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[i], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')


    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.savefig('/home/erschultz/downsampling_analysis/downsampling_figure_exp.png')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/downsampling_figure_exp.png')

    plt.close()




if __name__ == '__main__':
    mp.set_start_method('spawn')
    # make_samples()
    # fit_max_ent()
    # fit_gnn(631)
    figure(631)
