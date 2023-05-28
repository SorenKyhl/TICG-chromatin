import json
import os
import os.path as osp
import string
import sys

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

sys.path.append('/home/erschultz')

from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder


def run_long_simulation():
    if not osp.exists('/home/erschultz/downsampling_analysis/'):
        os.mkdir('/home/erschultz/downsampling_analysis')
    root = '/home/erschultz/downsampling_analysis/long_simulation'

    # dir2 = '/home/erschultz/dataset_02_04_23/samples/sample212/optimize_grid_b_140_phi_0.03-max_ent'
    dir = '/home/erschultz/dataset_04_28_23/samples/sample324'
    config = utils.load_json(osp.join(dir, 'config.json'))
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

def split_long_simulation():
    dir = '/home/erschultz/downsampling_analysis'
    production_dir = osp.join(dir, 'long_simulation3/production_out')
    for i in [1/3]:
    # [1, 2, 3, 4, 5, 10, 25, 50, 75, 100]:
        y = np.loadtxt(osp.join(production_dir, f'contacts{int(300000 * i/100)}.txt'))
        print(i)
        odir = osp.join(dir, f'sample{i}')
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

def analysis():
    dir = '/home/erschultz/downsampling_analysis/samples2'
    GNN_ID = 403
    gnn_scc = []
    max_ent_scc = []
    samples = np.array([1, 2, 3, 4, 5, 10, 25, 50, 75, 100])
    scc = SCC()
    y = np.load(osp.join(dir, 'sample100/y.npy'))
    y /= np.mean(y.diagonal())
    y_list = []
    yhat_list = []
    composites = []
    for i in samples:
        s_dir = osp.join(dir, f'sample{i}')
        y_i = np.load(osp.join(s_dir, 'y.npy'))
        y_i /= np.mean(y_i.diagonal())
        y_list.append(y_i)

        max_ent_dir = osp.join(s_dir, 'optimize_grid_b_140_phi_0.03-max_ent')
        final = get_final_max_ent_folder(max_ent_dir)
        # dist_pearson = utils.load_json(osp.join(final, 'distance_pearson.json'))
        # max_ent_scc.append(dist_pearson['scc_var'])
        yhat = np.load(osp.join(final, 'y.npy'))
        corr_scc_var = scc.scc(y, yhat, var_stabilized = True)
        max_ent_scc.append(corr_scc_var)

        gnn_dir = osp.join(s_dir, f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}')
        # dist_pearson = utils.load_json(osp.join(gnn_dir, 'distance_pearson.json'))
        # gnn_scc.append(dist_pearson['scc_var'])
        yhat = np.load(osp.join(gnn_dir, 'y.npy'))
        yhat_list.append(yhat)
        corr_scc_var = scc.scc(y, yhat, var_stabilized = True)
        gnn_scc.append(corr_scc_var)

        # make composite contact map
        m = len(y)
        indu = np.triu_indices(m)
        indl = np.tril_indices(m)
        composite = np.zeros((m, m))
        composite[indu] = yhat[indu]
        composite[indl] = y_i[indl]
        composites.append(composite)

    composites = np.array(composites)
    gnn_scc = np.array(gnn_scc)


    plt.plot(samples, max_ent_scc, label='Max Ent', color='blue')
    plt.plot(samples, gnn_scc, label='GNN', color='red')
    plt.xlabel('Downsampling %', fontsize=16)
    plt.xscale('log')
    plt.ylabel('SCC', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/erschultz/downsampling_analysis/scc.png')
    plt.close()

    ### plot hic ###
    ind = [0, 5, 9]
    fig, ax = plt.subplots(1, len(ind)+1, gridspec_kw={'width_ratios':[1,1,1,0.08]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    vmax = np.mean(composites)
    for i, (composite, scc, label) in enumerate(zip(composites[ind], gnn_scc[ind], samples[ind])):
        print(i)
        scc = np.round(scc, 3)
        if i == len(ind)-1:
            s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                            ax = ax[i], cbar_ax = ax[i+1])
        else:
            s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                            ax = ax[i], cbar = False)
        s.set_title(f'{label}% Sampling\nSCC={scc}', fontsize = 16)
        ax[i].axline((0,0), slope=1, color = 'k', lw=1)
        ax[i].text(0.99*m, 0.01*m, 'GNN', fontsize=16, ha='right', va='top')
        ax[i].text(0.01*m, 0.99*m, 'Reference', fontsize=16)

        if i > 0:
            s.set_yticks([])
    plt.tight_layout()
    plt.savefig('/home/erschultz/downsampling_analysis/hic.png')
    plt.close()

    ### combined plot ##
    # hic
    ind = [0, 5, 9]
    fig, axes = plt.subplots(1, len(ind)+2, gridspec_kw={'width_ratios':[1,1,1,0.01,1.5]})
    fig.set_figheight(6)
    fig.set_figwidth(6*3.5)
    vmax = np.mean(composites)
    for i, (composite, scc, label) in enumerate(zip(composites[ind], gnn_scc[ind], samples[ind])):
        print(i)
        scc = np.round(scc, 3)
        # if i == len(ind)-1:
        #     s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
        #                     ax = axes[i], cbar_ax = axes[i+1])
        # else:
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = axes[i], cbar = False)
        s.set_title(f'{label}% Sampling\nSCC={scc}', fontsize = 16)
        axes[i].axline((0,0), slope=1, color = 'k', lw=1)
        axes[i].text(0.99*m, 0.01*m, 'GNN', fontsize=16, ha='right', va='top')
        axes[i].text(0.01*m, 0.99*m, 'Reference', fontsize=16)

        if i > 0:
            s.set_yticks([])

    # filler subplot
    axes[-2].axis('off')

    # scc
    axes[-1].plot(samples, max_ent_scc, label='Max Ent', color='blue')
    axes[-1].plot(samples, gnn_scc, label='GNN', color='red')
    axes[-1].set_xlabel('Downsampling %', fontsize=16)
    axes[-1].set_xscale('log')
    axes[-1].set_ylabel('SCC', fontsize=16)
    axes[-1].legend()

    for n, ax in enumerate([axes[0], axes[-1]]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes,
                size=20, weight='bold')

    plt.tight_layout(h_pad=-0.5)
    plt.savefig('/home/erschultz/downsampling_analysis/downsampling_figure.png')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/downsampling_figure.png')

    plt.close()




if __name__ == '__main__':
    # run_long_simulation()
    # split_long_simulation()
    analysis()
    # smooth_samples()
