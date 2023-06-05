import json
import os
import os.path as osp
import string
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from pylib.utils.plotting_utils import RED_CMAP, plot_matrix
from pylib.utils.similarity_measures import SCC

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder
from sequences_to_contact_maps.scripts.utils import triu_to_full


def make_samples():
    exponents = np.arange(4, 9)
    dir = '/home/erschultz/downsampling_analysis'
    for exponent in exponents:
        e_dir = f'{dir}/samples_exp{exponent}'
        if not osp.exists(e_dir):
            os.mkdir(e_dir, mode=0o755)

    for s_exp in range(201, 211):
        exp_dir = f'/home/erschultz/dataset_02_04_23/samples/sample{s_exp}'
        y = np.triu(np.load(osp.join(exp_dir, 'y.npy')))
        p = y / np.sum(y)
        m = len(y)
        p_flat = p[np.triu_indices(m)]
        pos = np.arange(0, len(p_flat))

        for exponent in exponents:
            e_dir = f'{dir}/samples_exp{exponent}'
            print(exponent)
            odir = osp.join(e_dir, f'sample{s_exp}')
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

def figure():
    samples = np.arange(201, 211)
    N = len(samples)
    exponents = np.arange(4, 9)
    read_counts = 10**exponents
    dir = '/home/erschultz/downsampling_analysis'
    GNN_ID = 403
    m=512
    gnn_scc = defaultdict(lambda: np.zeros(N)) # exponent : list of sccs
    max_ent_scc = defaultdict(lambda: np.zeros(N))
    composite_dict = defaultdict(lambda: np.zeros((N,m,m))) # exponent: list of gnn composites

    y_list = [] # ground truth y from highly sampled simulation
    for s in samples:
        y = np.load(osp.join(dir, f'samples_exp8/sample{s}/y.npy'))
        y /= np.mean(y.diagonal())
        y_list.append(y)

    # collect data
    scc = SCC()
    for exp in exponents:
        for i, s in enumerate(samples):
            s_dir = osp.join(dir, f'samples_exp{exp}/sample{s}')
            y_sparse = np.load(osp.join(s_dir, 'y.npy'))
            y_sparse /= np.mean(y_sparse.diagonal())
            y_dense = y_list[i]

            max_ent_dir = osp.join(s_dir, 'optimize_grid_b_140_phi_0.03-max_ent10')
            final = get_final_max_ent_folder(max_ent_dir)
            yhat = np.load(osp.join(final, 'y.npy'))
            corr_scc_var = scc.scc(y_dense, yhat, var_stabilized = True)
            max_ent_scc[exp][i] = corr_scc_var

            gnn_dir = osp.join(s_dir, f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}')
            yhat = np.load(osp.join(gnn_dir, 'y.npy'))
            corr_scc_var = scc.scc(y_dense, yhat, var_stabilized = True)
            gnn_scc[exp][i] = corr_scc_var

            # make composite contact map
            m = len(y)
            indu = np.triu_indices(m)
            indl = np.tril_indices(m)
            composite = np.zeros((m, m))
            composite[indu] = yhat[indu]
            composite[indl] = y_sparse[indl]
            composite_dict[exp][i] = composite

    # compute statistics
    gnn_mean = np.zeros_like(exponents, dtype=float)
    max_ent_mean = np.zeros_like(exponents, dtype=float)
    gnn_std = np.zeros_like(exponents, dtype=float)
    max_ent_std = np.zeros_like(exponents, dtype=float)
    for i, exp in enumerate(exponents):
        gnn_mean[i] = np.mean(gnn_scc[exp])
        max_ent_mean[i] = np.mean(max_ent_scc[exp])

        gnn_std[i] = np.std(gnn_scc[exp])
        max_ent_std[i] = np.std(max_ent_scc[exp])

    ### combined plot ##
    # select which composites to plot
    sample_i = 4
    sample = samples[sample_i]
    exponents_hic = [5, 6, 7]

    fig, axes = plt.subplots(1, len(exponents_hic)+2, gridspec_kw={'width_ratios':[1,1,1,0.01,1.5]})
    fig.set_figheight(6)
    fig.set_figwidth(6*3.5)

    # hic
    composites = np.zeros((len(exponents_hic), m, m))
    for i, exp in enumerate(exponents_hic):
        composites[i] = composite_dict[exp][sample_i]
    vmax = np.mean(composites)
    for i, (exp, composite) in enumerate(zip(exponents_hic, composites)):
        read_count = 10**exp
        scc = gnn_scc[exp][sample_i]
        scc = np.round(scc, 3)
        # if i == len(ind)-1:
        #     s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
        #                     ax = axes[i], cbar_ax = axes[i+1])
        # else:
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = axes[i], cbar = False)
        s.set_title(f'Sample {sample}\nRead Count = 10^{exp}\nSCC={scc}', fontsize = 16)
        axes[i].axline((0,0), slope=1, color = 'k', lw=1)
        axes[i].text(0.99*m, 0.01*m, 'GNN', fontsize=16, ha='right', va='top')
        axes[i].text(0.01*m, 0.99*m, 'Reference', fontsize=16)

        if i > 0:
            s.set_yticks([])

    # filler subplot
    axes[-2].axis('off')

    # scc
    axes[-1].plot(read_counts, max_ent_mean, label='Max Ent', color='blue')
    axes[-1].fill_between(read_counts, max_ent_mean - max_ent_std, max_ent_mean + max_ent_std, color='blue', alpha=0.5)

    axes[-1].plot(read_counts, gnn_mean, label='GNN', color='red')
    axes[-1].fill_between(read_counts, gnn_mean - gnn_std, gnn_mean + gnn_std, color='red', alpha=0.5)

    axes[-1].set_xlabel('Read Count', fontsize=16)
    axes[-1].set_xscale('log')
    axes[-1].set_ylabel('SCC', fontsize=16)
    axes[-1].legend()

    for n, ax in enumerate([axes[0], axes[-1]]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes,
                size=20, weight='bold')

    plt.tight_layout(h_pad=-0.5)
    plt.savefig('/home/erschultz/downsampling_analysis/downsampling_figure_exp.png')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/downsampling_figure_exp.png')

    plt.close()




if __name__ == '__main__':
    # make_samples()
    figure()
