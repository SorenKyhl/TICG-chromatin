import math
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from modify_maxent import get_samples
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import BLUE_RED_CMAP, plot_matrix
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import make_composite
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz/TICG-chromatin/scripts')
from distances_Su2020.su2020_analysis import plot_diagonal


def low_rank_projections():
    dataset='dataset_02_04_23'
    data_dir = osp.join('/home/erschultz', dataset)
    samples, _ = get_samples(dataset, train=True)

    for sample in samples[:1]:
        s_dir = osp.join(data_dir, f'samples/sample{sample}')
        y_diag = np.load(osp.join(s_dir, 'y_diag.npy'))
        plot_matrix(y_diag, osp.join(s_dir, f'y_diag.png'),
                    vmin = 'center1', cmap = BLUE_RED_CMAP)

        # for i in [1,2,5,10,15,100]:
        #     # get y_diag top i PCs
        #     pca = PCA(n_components = i)
        #     y_transform = pca.fit_transform(y_diag)
        #     y_i = pca.inverse_transform(y_transform)
        #     rmse = np.round(mean_squared_error(y_i, y_diag, squared = False), 3)
        #     plot_matrix(y_i, osp.join(s_dir, f'y_diag_rank_{i}.png'),
        #                 vmin = 'center1', cmap = BLUE_RED_CMAP,
        #                 title = f'Y_diag rank {i}, RMSE={rmse}')

        y_corr = np.corrcoef(y_diag)
        plot_matrix(y_corr, osp.join(s_dir, f'y_corr.png'),
                    vmin = 'center', cmap = BLUE_RED_CMAP)
        # for i in [1,2,5,10,15,100]:
        #     # get y_diag top i PCs
        #     pca = PCA(n_components = i)
        #     y_transform = pca.fit_transform(y_corr)
        #     y_i = pca.inverse_transform(y_transform)
        #     rmse = np.round(mean_squared_error(y_i, y_corr, squared = False), 3)
        #     plot_matrix(y_i, osp.join(s_dir, f'y_corr_rank_{i}.png'),
        #                 vmin = 'center', cmap = BLUE_RED_CMAP,
        #                 title = f'Y_corr rank {i}, RMSE={rmse}')



def compare_degree_distribution():
    dataset='dataset_02_04_23'
    data_dir = osp.join('/home/erschultz', dataset)
    data_dir_max_ent = data_dir + '_max_ent'
    samples, _ = get_samples(dataset, train=True)

    for sample in samples[:1]:
        s_dir = osp.join(data_dir, f'samples/sample{sample}')
        s_dir_max_ent = osp.join(data_dir_max_ent, f'samples/sample{sample}')
        assert osp.exists(s_dir_max_ent)

        y_exp = np.load(osp.join(s_dir, 'y.npy'))
        y_max_ent = np.load(osp.join(s_dir_max_ent, 'y.npy'))

        bin_width = 0.4
        for arr, label in zip([y_exp, y_max_ent], ['Experiment', 'Max Ent']):
            arr /= np.mean(np.diagonal(arr))
            arr = arr[np.triu_indices(len(arr), 1)]
            arr = arr[arr>0]
            arr = np.log(arr)
            print(np.min(arr))
            print(arr)
            bin_positions = np.arange(math.floor(min(arr)),
                                    math.ceil(max(arr)) + bin_width,
                                    bin_width)
            _, bins, _ = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins=bin_positions,
                                    alpha = 0.5, label = label)

        plt.legend()
        plt.ylabel('probability', fontsize=16)
        plt.xlabel(r'$Y_{ij}$', fontsize=16)
        # plt.xscale('log')
        plt.tight_layout()
        # plt.show()
        plt.savefig(osp.join(s_dir_max_ent, 'deg_dist.png'))
        plt.close()

def compare_tads():
    dataset='dataset_02_04_23'
    data_dir = osp.join('/home/erschultz', dataset)
    data_dir_max_ent = data_dir + '_max_ent'
    samples, _ = get_samples(dataset, train=True)

    for sample in samples[:1]:
        s_dir = osp.join(data_dir, f'samples/sample{sample}')
        s_dir_max_ent = osp.join(data_dir_max_ent, f'samples/sample{sample}')
        assert osp.exists(s_dir_max_ent)

        y_exp = np.load(osp.join(s_dir, 'y.npy'))
        y_max_ent = np.load(osp.join(s_dir_max_ent, 'y.npy'))
        plot_diagonal(y_exp, y_max_ent, osp.join(s_dir_max_ent, 'tads.png'))
        plot_matrix(make_composite(y_exp, y_max_ent),
                    osp.join(s_dir_max_ent, 'composite.png'),
                    vmax = 'mean', triu=True)
        plot_matrix(make_composite(y_exp, y_max_ent),
                    osp.join(s_dir_max_ent, 'composite_light.png'),
                    vmax = np.percentile(y_exp, 95), triu=True)

        y_diag_exp = np.load(osp.join(s_dir, 'y_diag.npy'))
        y_diag_max_ent = np.load(osp.join(s_dir_max_ent, 'y_diag.npy'))
        plot_diagonal(y_diag_exp, y_diag_max_ent, osp.join(s_dir_max_ent, 'diag_tads.png'))
        plot_matrix(make_composite(y_diag_exp, y_diag_max_ent),
                    osp.join(s_dir_max_ent, 'diag_composite.png'),
                    vmin = 'center1', cmap = BLUE_RED_CMAP, triu=True)

        y_corr_exp = np.corrcoef(y_diag_exp)
        y_corr_max_ent = np.corrcoef(y_diag_max_ent)
        plot_diagonal(y_corr_exp, y_corr_max_ent, osp.join(s_dir_max_ent, 'corr_tads.png'))
        plot_matrix(make_composite(y_corr_exp, y_corr_max_ent),
                    osp.join(s_dir_max_ent, 'corr_composite.png'),
                    vmin = 'center', cmap = BLUE_RED_CMAP, triu=True)


def compare_SCC_exp_vs_exp_max_ent(GNN_ID):
    '''Assess GNN performance when experiment is input or max ent simulation is input.'''
    dataset='dataset_02_04_23'
    data_dir = osp.join('/home/erschultz', dataset)
    data_dir_max_ent = data_dir + '_max_ent'
    assert osp.exists(data_dir_max_ent)
    samples, _ = get_samples(dataset, train=True)
    grid_root = 'optimize_grid_b_180_phi_0.008_spheroid_1.5'
    GNN_root = f'{grid_root}-GNN{GNN_ID}'
    max_ent_root =  f'{grid_root}-max_ent5'
    scc = SCC()

    for sample in samples[:1]:
        s_dir = osp.join(data_dir, f'samples/sample{sample}')
        s_dir_max_ent = osp.join(data_dir_max_ent, f'samples/sample{sample}')
        assert osp.exists(s_dir_max_ent)

        y_exp = np.load(osp.join(s_dir, 'y_diag.npy'))

        y_max_ent = np.load(osp.join(s_dir_max_ent, 'y_diag.npy'))
        y_max_ent_max_ent = np.load(osp.join(s_dir_max_ent, max_ent_root, 'iteration30/y.npy'))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_max_ent_max_ent)
        y_max_ent_max_ent = DiagonalPreprocessing.process(y_max_ent_max_ent, meanDist, verbose = False)

        y_exp_gnn = np.load(osp.join(s_dir, GNN_root, 'y.npy'))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_exp_gnn)
        y_exp_gnn = DiagonalPreprocessing.process(y_exp_gnn, meanDist, verbose = False)

        y_max_ent_gnn = np.load(osp.join(s_dir_max_ent, GNN_root, 'y.npy'))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_max_ent_gnn)
        y_max_ent_gnn = DiagonalPreprocessing.process(y_max_ent_gnn, meanDist, verbose = False)


        corr_scc_var = scc.scc(y_exp, y_max_ent, var_stabilized = True)
        print(f'exp, exp_max_ent: {corr_scc_var}')
        corr_scc_var = scc.scc(y_exp, y_exp_gnn, var_stabilized = True)
        print(f'exp, exp_GNN: {corr_scc_var}')
        corr_scc_var = scc.scc(y_max_ent, y_max_ent_gnn, var_stabilized = True)
        print(f'exp_max_ent, max_ent_GNN: {corr_scc_var}')
        corr_scc_var = scc.scc(y_max_ent, y_max_ent_max_ent, var_stabilized = True)
        print(f'exp_max_ent, max_ent_max_ent: {corr_scc_var}')
        corr_scc_var = scc.scc(y_exp, y_max_ent_gnn, var_stabilized = True)
        print(f'exp, max_ent_GNN: {corr_scc_var}')
        corr_scc_var = scc.scc(y_exp, y_max_ent_max_ent, var_stabilized = True)
        print(f'exp, max_ent_max_ent: {corr_scc_var}')

if __name__ == '__main__':
    # compare_degree_distribution()
    # compare_tads()
    low_rank_projections()
    # compare_SCC_exp_vs_exp_max_ent(519)
