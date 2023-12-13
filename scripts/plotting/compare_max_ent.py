import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pylib.utils import epilib
# from data_generation.modify_maxent import get_samples
from pylib.utils.plotting_utils import BLUE_RED_CMAP, RED_CMAP, plot_matrix
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import make_composite
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder


def compare_sequences():
    dir_exp = '/home/erschultz/dataset_11_20_23/samples/sample8'
    dir = osp.join(dir_exp, 'optimize_grid_b_180_v_8_spheroid_1.5-max_ent9_chrom_smooth/iteration2')
    odir = osp.split(dir)[0]
    y_gt = np.load(osp.join(dir_exp, 'y.npy'))
    y_gt /= np.mean(y_gt.diagonal())
    np.fill_diagonal(y_gt, 1)
    y_me = np.load(osp.join(dir, 'y.npy'))
    lines = [245, 360]
    y = make_composite(y_gt, y_me)
    plot_matrix(y, osp.join(odir, 'y.png'), vmax='mean', triu=True, lines=lines)


    def plot_seq(seq, label):
        m, k = seq.shape
        print(seq.shape)
        fig, axes = plt.subplots(2, 3)
        fig.set_figheight(6)
        fig.set_figwidth(12)
        for i, ax in enumerate(axes.flatten()):
            if i >= k:
                continue
            ax.plot(np.arange(0, m), seq[:, i])

            for line in lines:
                ax.axvline(line, c='k', lw=0.5)

        fig.supxlabel('Distance', fontsize=16)
        fig.supylabel('Label Value', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(odir, label))
        plt.close()

    x = np.load(osp.join(dir, 'x.npy')).T
    plot_seq(x,'seq_actual.png')

    k=5
    # x_repeat = epilib.get_pcs(epilib.get_oe(y_gt), k, normalize = True)
    # plot_seq(x_repeat,'seq_repeat.png')
    #
    # x_smooth = epilib.get_pcs(epilib.get_oe(y_gt), k, normalize = True,
    #                             smooth=True, h=1)
    # plot_seq(x_smooth,'seq_smooth1.png')
    #
    # x_smooth3 = epilib.get_pcs(epilib.get_oe(y_gt), k, normalize = True,
    #                             smooth=True, h=3)
    # plot_seq(x_smooth3,'seq_smooth3.png')
    #
    # x_corr = epilib.get_pcs(np.corrcoef(epilib.get_oe(y_gt)), k, normalize = True)
    # plot_seq(x_corr,'seq_corr.png')
    #
    # x_corr = epilib.get_pcs(np.corrcoef(epilib.get_oe(y_gt)), k, normalize = False)
    # plot_seq(x_corr,'seq_corr_no_norm.png')

    x_soren = epilib.get_sequences(y_gt, k).T
    plot_seq(x_soren, 'seq_soren.png')


def compare_methods(dataset):
    dir = f'/home/erschultz/{dataset}/samples'
    odir = f'/home/erschultz/{dataset}/figures'
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    max_ent_roots = ['optimize_grid_b_200_v_8_spheroid_1.5-max_ent10']
                    # 'optimize_grid_b_180_v_8_spheroid_1.5-max_ent10_long',
                    # 'optimize_grid_b_180_v_8_spheroid_1.5-GNN614',
                    # ]
    names = ['Baseline']
    # , 'Long', 'GNN 614']
    # samples = get_samples(dataset, test=True, filter_cell_lines='imr90')
    # samples = [1,2,3,4,5,13,14,15]
    samples = [6, 7, 8, 9, 10, 11, 12]

    scc = SCC(h=5, K=100)

    for sample in samples:
        s_dir = osp.join(dir, f'sample{sample}')
        y_gt = np.load(osp.join(s_dir, 'y.npy'))
        y_gt /= np.mean(y_gt.diagonal())
        y_diag_gt = epilib.get_oe(y_gt)

        pcs_gt = epilib.get_pcs(epilib.get_oe(y_gt), 12).T


        fig, axes = plt.subplots(1, len(names))
        if len(names) == 1:
            axes = [axes]
        fig.set_figheight(6)
        fig.set_figwidth(14/3*len(names))

        for i, (max_ent_root, ax, name) in enumerate(zip(max_ent_roots, axes, names)):
            me_dir = osp.join(s_dir, max_ent_root)
            if name.startswith('GNN'):
                final = me_dir
            else:
                final = get_final_max_ent_folder(me_dir)
            y_me = np.load(osp.join(final, 'y.npy'))
            y = make_composite(y_gt, y_me)
            m = len(y)

            scc_var = scc.scc(y_gt, y_me, var_stabilized = True)
            scc_var = np.round(scc_var, 3)

            pcs_me = epilib.get_pcs(epilib.get_oe(y_me), 12).T
            pearson_pc_1, _ = pearsonr(pcs_me[0], pcs_gt[0])
            pearson_pc_1 *= np.sign(pearson_pc_1) # ensure positive pearson
            assert pearson_pc_1 > 0
            pearson_pc_1 = np.round(pearson_pc_1, 3)

            y_diag_me = epilib.get_oe(y_me)
            rmse_y_tilde = mean_squared_error(y_diag_gt, y_diag_me, squared=False)
            rmse_y_tilde = np.round(rmse_y_tilde, 3)


            title = f'SCC={scc_var}\nCorr PC 1={pearson_pc_1}\n'+r'RMSE($\tilde{H}$)'+f'={rmse_y_tilde}'

            s = sns.heatmap(y, linewidth = 0, vmin = 0, vmax = np.mean(y_gt), cmap = RED_CMAP,
                            ax = ax, cbar = False)
            s.set_title(title, fontsize = 16, loc='left')

            ax.axline((0,0), slope=1, color = 'k', lw=1)
            s.text(0.99*m, -0.08*m, name, fontsize=16, ha='right', va='top', weight='bold')
            s.text(0.01*m, 1.08*m, 'Experiment', fontsize=16, weight='bold')

            s.set_xticks([])
            if i > 0:
                s.set_yticks([])
            else:
                s.set_yticks([0, 100, 512], [0, 100, 512], fontsize=16)

        plt.tight_layout()
        plt.savefig(osp.join(odir, f'max_ent_comparison_sample{sample}.png'))
        plt.close()

if __name__ == '__main__':
    compare_methods('dataset_12_06_23')
