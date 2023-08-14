import json
import multiprocessing
import os
import os.path as osp
import string
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from pylib import analysis
from pylib.Pysim import Pysim
from pylib.utils import utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import RED_CMAP, plot_matrix, plot_mean_dist
from pylib.utils.similarity_measures import SCC
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_max_ent_S, load_S)


def run_long_simulation_wrapper():
    samples = range(201, 211)
    odir = '/home/erschultz/downsampling_analysis/samples_long'
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)

    wrapper = []
    for sample in samples:
        wrapper.append((sample, osp.join(odir, f'sample{sample}')))

    with multiprocessing.Pool(11) as p:
        p.starmap(run_long_simulation, wrapper)

def run_long_simulation(sample, root = '/home/erschultz/downsampling_analysis/long_simulation'):
    if not osp.exists('/home/erschultz/downsampling_analysis/'):
        os.mkdir('/home/erschultz/downsampling_analysis')

    dir = f'/home/erschultz/dataset_02_04_23/samples/sample{sample}/optimize_grid_b_140_phi_0.03-max_ent10'
    # dir = '/home/erschultz/dataset_04_28_23/samples/sample324'
    final = get_final_max_ent_folder(dir)
    config = utils.load_json(osp.join(final, 'config.json'))
    config['bead_type_files'] = [f'pcf{i}.txt' for i in range(1, config['nspecies']+1)]
    config['track_contactmap'] = True
    config['dump_frequency'] = 1000
    # k = config['nspecies']
    # chis = np.zeros((k, k))
    # LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # for i in range(k):
    #     chis[i,i] = config[f'chi{LETTERS[i]}{LETTERS[i]}']
    # config['chis'] = chis.tolist()
    # config['profiling_on'] = False

    # get sequences
    seqs = np.load(osp.join(dir, 'x.npy'))

    sim = Pysim(root, config, seqs, None, randomize_seed = False, overwrite = True)
    sim.run_eq(50000, 300000, 1)

    with utils.cd(sim.root):
        analysis.main_no_maxent()

def split_long_simulation_wrapper():
    samples = range(201, 211)

    with multiprocessing.Pool(11) as p:
        p.map(split_long_simulation, samples)

def split_long_simulation(sample):
    dir = '/home/erschultz/downsampling_analysis'

    sampling_rates = [1, 5, 10, 25, 50, 75, 100]
    for i in sampling_rates:
        short_dir = osp.join(dir, f'samples_sim{i}')
        if not osp.exists(short_dir):
            os.mkdir(short_dir, mode=0o755)

    long_dir = osp.join(dir, f'samples_long/sample{sample}')
    production_dir = osp.join(long_dir, 'production_out')
    for i in sampling_rates:
        short_dir = osp.join(dir, f'samples_sim{i}')
        y = np.loadtxt(osp.join(production_dir, f'contacts{int(300000 * i/100)}.txt'))
        odir = osp.join(short_dir, f'sample{sample}')
        if not osp.exists(odir):
            os.mkdir(odir)
        np.save(osp.join(odir, 'y.npy'), y)
        plot_matrix(y, osp.join(odir, 'y.png'), vmax = 'mean')

def smooth_samples():
    dir = '/home/erschultz/downsampling_analysis/samples2'
    odir = dir + '_smooth'
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    for f in os.listdir(dir):
        y = np.load(osp.join(dir, f, 'y.npy'))
        # y_smooth = scipy.ndimage.gaussian_filter(y, 1)
        y_smooth = scipy.ndimage.uniform_filter(y, 3)
        fdir = osp.join(odir, f)
        if not osp.exists(fdir):
            os.mkdir(fdir, mode=0o755)
        np.save(osp.join(fdir, 'y.npy'), y_smooth)
        plot_matrix(y_smooth, osp.join(odir, f, 'y.png'), vmax = 'mean')

def figure():
    samples = np.arange(201, 211)
    N = len(samples)
    downsamplings = [1,5,10,25,50,75,100]
    dir = '/home/erschultz/downsampling_analysis'
    GNN_ID = 403
    m=512
    gnn_scc = defaultdict(lambda: np.zeros(N)) # downsampling : list of sccs
    gnn_mse = defaultdict(lambda: np.zeros(N)) # downsampling : list of mses
    max_ent_scc = defaultdict(lambda: np.zeros(N))
    max_ent_mse = defaultdict(lambda: np.zeros(N))
    composite_dict = defaultdict(lambda: np.zeros((N,m,m))) # downsampling: list of gnn composites

    y_list = [] # ground truth y from highly sampled simulation
    S_list = []
    for s in samples:
        long_dir = osp.join(dir, f'samples_long/sample{s}')
        y = np.loadtxt(osp.join(long_dir, 'production_out/contacts300000.txt'))
        y /= np.mean(y.diagonal())
        y_list.append(y)

        S = load_S(long_dir)
        S_list.append(S)


    # collect data
    scc = SCC()
    for d in downsamplings:
        for i, s in enumerate(samples):
            s_dir = osp.join(dir, f'samples_sim{d}/sample{s}')
            y_sparse = np.load(osp.join(s_dir, 'y.npy'))
            y_sparse /= np.mean(y_sparse.diagonal())
            y_dense = y_list[i]
            S_gt = S_list[i]

            max_ent_dir = osp.join(s_dir, 'optimize_grid_b_140_phi_0.03-max_ent10')
            if osp.exists(max_ent_dir):
                final = get_final_max_ent_folder(max_ent_dir)
                y_file = osp.join(final, 'y.npy')
                if osp.exists(y_file):
                    yhat = np.load(y_file)
                    corr_scc_var = scc.scc(y_dense, yhat, var_stabilized = True)
                    max_ent_scc[d][i] = corr_scc_var

                    S = load_max_ent_S(max_ent_dir, True)
                    mse = mean_squared_error(S_gt, S)
                    max_ent_mse[d][i] = mse



            gnn_dir = osp.join(s_dir, f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}')
            y_file = osp.join(gnn_dir, 'y.npy')
            if osp.exists(y_file):
                yhat = np.load(y_file)
            else:
                print(f'WARNING: {y_file} is missing')
                continue
            corr_scc_var = scc.scc(y_dense, yhat, var_stabilized = True)
            gnn_scc[d][i] = corr_scc_var

            S = np.load(osp.join(gnn_dir, 'S.npy'))
            mse = mean_squared_error(S_gt, S)
            gnn_mse[d][i] = mse

            # make composite contact map
            m = len(y)
            indu = np.triu_indices(m)
            indl = np.tril_indices(m)
            composite = np.zeros((m, m))
            composite[indu] = yhat[indu]
            composite[indl] = y_sparse[indl]
            composite_dict[d][i] = composite

    # compute statistics
    gnn_mean = np.zeros_like(downsamplings, dtype=float)
    gnn_std = np.zeros_like(downsamplings, dtype=float)
    max_ent_mean = np.zeros_like(downsamplings, dtype=float)
    max_ent_std = np.zeros_like(downsamplings, dtype=float)
    gnn_mse_mean = np.zeros_like(downsamplings, dtype=float)
    gnn_mse_std = np.zeros_like(downsamplings, dtype=float)
    max_ent_mse_mean = np.zeros_like(downsamplings, dtype=float)
    max_ent_mse_std = np.zeros_like(downsamplings, dtype=float)
    for i, d in enumerate(downsamplings):
        gnn_mean[i] = np.mean(gnn_scc[d])
        gnn_std[i] = np.std(gnn_scc[d])

        max_ent_mean[i] = np.mean(max_ent_scc[d])
        max_ent_std[i] = np.std(max_ent_scc[d])

        gnn_mse_mean[i] = np.mean(gnn_mse[d])
        gnn_mse_std[i] = np.std(gnn_mse[d])

        max_ent_mse_mean[i] = np.mean(max_ent_mse[d])
        max_ent_mse_std[i] = np.std(max_ent_mse[d])


    ### combined plot ##
    # select which composites to plot
    sample_i = 4
    sample = samples[sample_i]
    downsamplings_hic = [10, 25, 75]

    fig, axes = plt.subplots(1, len(downsamplings_hic)+2,
                            gridspec_kw={'width_ratios':[1,1,1,0.01,1.5]})
    fig.set_figheight(6)
    fig.set_figwidth(6*3.5)

    # hic
    composites = np.zeros((len(downsamplings_hic), m, m))
    for i, d in enumerate(downsamplings_hic):
        composites[i] = composite_dict[d][sample_i]
    vmax = np.mean(composites)
    for i, (d, composite) in enumerate(zip(downsamplings_hic, composites)):
        scc = gnn_scc[d][sample_i]
        scc = np.round(scc, 3)
        # if i == len(ind)-1:
            # s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax,
            #                 cmap = RED_CMAP,
            #                 ax = axes[i], cbar_ax = axes[i+1])
        # else:
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = axes[i], cbar = False)
        s.set_title(f'Sample {sample}\nDownsampling = {d}\nSCC={scc}', fontsize = 16)
        axes[i].axline((0,0), slope=1, color = 'k', lw=1)
        axes[i].text(0.99*m, 0.01*m, 'GNN', fontsize=16, ha='right', va='top')
        axes[i].text(0.01*m, 0.99*m, 'Reference', fontsize=16)

        if i > 0:
            s.set_yticks([])

    # filler subplot
    axes[-2].axis('off')

    # scc
    axes[-1].plot(downsamplings, max_ent_mean, label='Max Ent', color='blue')
    axes[-1].fill_between(downsamplings, max_ent_mean - max_ent_std,
                        max_ent_mean + max_ent_std, color='blue', alpha=0.5)
    twinax = axes[-1].twinx()
    axes[-1].plot(downsamplings, gnn_mean, label='GNN', color='red')
    axes[-1].fill_between(downsamplings, gnn_mean - gnn_std, gnn_mean + gnn_std,
                        color='red', alpha=0.5)

    twinax.plot(downsamplings, gnn_mse_mean, label='GNN', color='red')
    twinax.fill_between(downsamplings, gnn_mse_mean - gnn_mse_std,
                        gnn_mse_mean + gnn_mse_std, color='red', alpha=0.5)

    axes[-1].set_xlabel('Downsampling', fontsize=16)
    # axes[-1].set_xscale('log')
    axes[-1].set_ylabel('SCC', fontsize=16)
    twinax.set_ylabel('MSE', fontsize=16)
    axes[-1].legend()

    for n, ax in enumerate([axes[0], axes[-1]]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes,
                size=20, weight='bold')

    plt.tight_layout(h_pad=-0.5)
    plt.savefig('/home/erschultz/downsampling_analysis/downsampling_figure_sim.png')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/downsampling_figure_sim.png')

    plt.close()

def figure_v2():
    samples = np.arange(201, 211)
    N = len(samples)
    downsamplings = [1,5,10,25,50,75,100]
    dir = '/home/erschultz/downsampling_analysis'
    GNN_ID = 403
    m=512
    gnn_scc = defaultdict(lambda: np.zeros(N)) # downsampling : list of sccs
    gnn_mse = defaultdict(lambda: np.zeros(N)) # downsampling : list of mses
    max_ent_scc = defaultdict(lambda: np.zeros(N))
    max_ent_mse = defaultdict(lambda: np.zeros(N))
    composite_dict = defaultdict(lambda: np.zeros((N,m,m))) # downsampling: list of gnn composites

    y_list = [] # ground truth y from highly sampled simulation
    S_list = []
    for s in samples:
        long_dir = osp.join(dir, f'samples_long/sample{s}')
        y = np.loadtxt(osp.join(long_dir, 'production_out/contacts300000.txt'))
        y /= np.mean(y.diagonal())
        y_list.append(y)

        S = load_S(long_dir)
        S_list.append(S)


    # collect data
    scc = SCC()
    for d in downsamplings:
        for i, s in enumerate(samples):
            s_dir = osp.join(dir, f'samples_sim{d}/sample{s}')
            y_sparse = np.load(osp.join(s_dir, 'y.npy'))
            y_sparse /= np.mean(y_sparse.diagonal())
            y_dense = y_list[i]
            S_gt = S_list[i]

            max_ent_dir = osp.join(s_dir, 'optimize_grid_b_140_phi_0.03-max_ent10')
            if osp.exists(max_ent_dir):
                final = get_final_max_ent_folder(max_ent_dir)
                y_file = osp.join(final, 'y.npy')
                if osp.exists(y_file):
                    yhat = np.load(y_file)
                    corr_scc_var = scc.scc(y_dense, yhat, var_stabilized = True)
                    max_ent_scc[d][i] = corr_scc_var

                    S = load_max_ent_S(max_ent_dir, True)
                    mse = mean_squared_error(S_gt, S)
                    max_ent_mse[d][i] = mse



            gnn_dir = osp.join(s_dir, f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}')
            y_file = osp.join(gnn_dir, 'y.npy')
            if osp.exists(y_file):
                yhat = np.load(y_file)
            else:
                print(f'WARNING: {y_file} is missing')
                continue
            corr_scc_var = scc.scc(y_dense, yhat, var_stabilized = True)
            gnn_scc[d][i] = corr_scc_var

            S = np.load(osp.join(gnn_dir, 'S.npy'))
            mse = mean_squared_error(S_gt, S)
            gnn_mse[d][i] = mse

            # make composite contact map
            m = len(y)
            indu = np.triu_indices(m)
            indl = np.tril_indices(m)
            composite = np.zeros((m, m))
            composite[indu] = yhat[indu]
            composite[indl] = y_sparse[indl]
            composite_dict[d][i] = composite

    # compute statistics
    gnn_mean = np.zeros_like(downsamplings, dtype=float)
    gnn_std = np.zeros_like(downsamplings, dtype=float)
    max_ent_mean = np.zeros_like(downsamplings, dtype=float)
    max_ent_std = np.zeros_like(downsamplings, dtype=float)
    gnn_mse_mean = np.zeros_like(downsamplings, dtype=float)
    gnn_mse_std = np.zeros_like(downsamplings, dtype=float)
    max_ent_mse_mean = np.zeros_like(downsamplings, dtype=float)
    max_ent_mse_std = np.zeros_like(downsamplings, dtype=float)
    for i, d in enumerate(downsamplings):
        gnn_mean[i] = np.mean(gnn_scc[d])
        gnn_std[i] = np.std(gnn_scc[d])

        max_ent_mean[i] = np.mean(max_ent_scc[d])
        max_ent_std[i] = np.std(max_ent_scc[d])

        gnn_mse_mean[i] = np.mean(gnn_mse[d])
        gnn_mse_std[i] = np.std(gnn_mse[d])

        max_ent_mse_mean[i] = np.mean(max_ent_mse[d])
        max_ent_mse_std[i] = np.std(max_ent_mse[d])


    ### combined plot ##
    # select which composites to plot
    sample_i = 4
    sample = samples[sample_i]
    downsamplings_hic = [10, 25, 75]

    plt.figure(figsize=(18, 12))
    ax1 = plt.subplot(2, 12, (1, 4))
    ax2 = plt.subplot(2, 12, (5, 8))
    ax3 = plt.subplot(2, 12, (9, 12))
    ax4 = plt.subplot(2, 12, (13, 18)) # scc
    ax5 = plt.subplot(2, 12, (19, 24)) # mse
    axes = [ax1, ax2, ax3, ax4, ax5]
    # hic
    composites = np.zeros((len(downsamplings_hic), m, m))
    for i, d in enumerate(downsamplings_hic):
        composites[i] = composite_dict[d][sample_i]
    vmax = np.mean(composites)
    for i, (d, composite) in enumerate(zip(downsamplings_hic, composites)):
        scc = gnn_scc[d][sample_i]
        scc = np.round(scc, 3)
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = axes[i], cbar = False)
        s.set_title(f'Sample {sample}\nDownsampling = {d}\nSCC={scc}', fontsize = 16)
        axes[i].axline((0,0), slope=1, color = 'k', lw=1)
        axes[i].text(0.99*m, 0.01*m, 'GNN', fontsize=16, ha='right', va='top')
        axes[i].text(0.01*m, 0.99*m, 'Reference', fontsize=16)

        if i > 0:
            s.set_yticks([])

    # scc
    ax4.plot(downsamplings, max_ent_mean, label='Max Ent', color='blue')
    ax4.fill_between(downsamplings, max_ent_mean - max_ent_std,
                        max_ent_mean + max_ent_std, color='blue', alpha=0.5)
    ax4.plot(downsamplings, gnn_mean, label='GNN', color='red')
    ax4.fill_between(downsamplings, gnn_mean - gnn_std, gnn_mean + gnn_std,
                        color='red', alpha=0.5)

    ax4.set_xlabel('Downsampling', fontsize=16)
    # ax4.set_xscale('log')
    ax4.set_ylabel('SCC', fontsize=16)
    ax4.legend()


    ax5.plot(downsamplings, gnn_mse_mean, label='GNN', color='red')
    ax5.fill_between(downsamplings, gnn_mse_mean - gnn_mse_std,
                        gnn_mse_mean + gnn_mse_std, color='red', alpha=0.5)
    ax5.plot(downsamplings, max_ent_mse_mean, label='Max Ent', color='blue')
    ax5.fill_between(downsamplings, max_ent_mse_mean - max_ent_mse_std,
                        max_ent_mse_mean + max_ent_mse_std, color='blue', alpha=0.5)

    ax5.set_xlabel('Downsampling', fontsize=16)
    # ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_ylabel('MSE', fontsize=16)
    ax5.legend()


    for n, ax in enumerate([ax1, ax4, ax5]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes,
                size=20, weight='bold')

    plt.tight_layout()
    plt.savefig('/home/erschultz/downsampling_analysis/downsampling_figure_sim.png')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/downsampling_figure_sim.png')

    plt.close()



if __name__ == '__main__':
    # run_long_simulation_wrapper()
    # split_long_simulation_wrapper()
    # figure()
    figure_v2()
    # smooth_samples()
