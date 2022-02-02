import sys
import os
import os.path as osp

import numpy as np

# ensure that I can find makeLatexTable
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from makeLatexTable import METHODS
from r_pca import R_pca

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.dataset_classes import make_dataset
from neural_net_utils.utils import *
from result_summary_plots import *

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def find_mising_ids():
    ids = set(range(1, 2001))
    dir = "/project2/depablo/erschultz/dataset_10_27_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            data_out_path = osp.join(dir, file, 'data_out')
            x_path = osp.join(dir, file, 'x.npy')
            if osp.exists(data_out_path) and osp.exists(x_path):
                ids.remove(id)

    print(ids, len(ids))

def upper_traingularize_chis():
    dir = "/project2/depablo/erschultz/dataset_08_26_21"
    samples = make_dataset(dir)
    for file in samples:
        file_dir = osp.join(dir, file)
        chis = np.load(osp.join(file_dir, 'chis.npy'))
        chis = np.triu(chis)

        np.savetxt(osp.join(file_dir, 'chis.txt'), chis, fmt='%0.5f')
        np.save(osp.join(file_dir, 'chis.npy'), chis)

def write_psi():
    # dir = "/project2/depablo/erschultz/dataset_11_03_21/samples"
    dir = "/home/eric/sequences_to_contact_maps/dataset_11_03_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            file_dir = osp.join(dir, file)

            xfile = osp.join(file_dir, 'x.npy')
            x = np.load(xfile)
            m, k = x.shape
            seq = np.zeros((m ,k))
            for i in range(k):
                seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                seq[:, i] = seq_i
            if not np.array_equal(seq, x):
                print(int(file[6:]))

            x_linear_file = osp.join(file_dir, 'x_linear.npy')
            if osp.exists(x_linear_file):
                x_linear = np.load(x_linear_file)
                np.save(osp.join(file_dir, 'psi.npy'), x_linear)

def check_seq():
    # dir = "/project2/depablo/erschultz/dataset_11_03_21/samples"
    dir = "/home/eric/sequences_to_contact_maps/dataset_11_03_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            file_dir = osp.join(dir, file)

            xfile = osp.join(file_dir, 'x.npy')
            x = np.load(xfile)
            m, _ = x.shape
            k = 4
            seq = np.zeros((m, k))
            for i in range(k):
                seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                seq[:, i] = seq_i
            # if not np.array_equal(seq, x):
            #     print(int(file[6:]))

            psi_file = osp.join(file_dir, 'psi.npy')
            if osp.exists(psi_file):
                psi = np.load(psi_file)
                if not np.array_equal(seq, psi):
                    print(psi)
                    print(seq)
                    print(int(file[6:]))

def makeDirsForMaxEnt(dataset, sample):
    sample_folder = osp.join('../sequences_to_contact_maps', dataset, 'samples', 'sample{}'.format(sample))
    assert osp.exists(sample_folder)

    for method in ['ground-truth', 'ground_truth-S', 'PCA', 'k_means', 'nmf', 'GNN-44-S']:
        os.mkdir(osp.join(sample_folder, method), mode = 0o755)
        for k in [2, 4]:
            os.mkdir(osp.join(sample_folder, method, f'k{k}'), mode = 0o755)
            for replicate in [1]:
                os.mkdir(osp.join(sample_folder, method, f'k{k}', f'replicate{replicate}'), mode = 0o755)

def main():
    dir = '/home/eric/dataset_test/samples/sample90'
    e = np.load(osp.join(dir, 'e.npy'))
    y = np.load(osp.join(dir, 'y.npy'))
    y_max = np.max(y)

    e1024 = e[:1024, :1024]
    print(e1024.shape)
    np.save(osp.join(dir, 'e1024.npy'), e1024)
    np.save(osp.join(dir, 'y1024.npy'), y[:1024, :1024])
    plotContactMap(y[:1024, :1024] / y_max, ofile = osp.join(dir, 'y1024_prob.png'), vmax = 'mean')

    e1024_2 = e[6000:7024, 6000:7024]
    print(e1024_2.shape)
    np.save(osp.join(dir, 'e1024_2.npy'), e1024_2)
    np.save(osp.join(dir, 'y1024_2.npy'), y[6000:7024, 6000:7024])
    plotContactMap(y[6000:7024, 6000:7024] / y_max, ofile = osp.join(dir, 'y1024_2_prob.png'), vmax = 'mean')

    dir = '/home/eric/dataset_test/samples/sample91'
    y = np.load(osp.join(dir, 'y.npy'))
    y_max = np.max(y)
    plotContactMap(y / y_max, ofile = osp.join(dir, 'y_prob.png'), vmax = 'mean')

def test_robust_PCA():
    dir = '/home/eric/dataset_test/samples/sample81'
    # dir = '/home/eric/sequences_to_contact_maps/dataset_09_21_21/samples/sample1'
    y = np.load(osp.join(dir, 'y.npy'))
    L, S = R_pca(y).fit(max_iter=500)
    plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = np.mean(y))
    plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmax = np.mean(y))

    y_diag = np.load(osp.join(dir, 'y_diag.npy'))
    L, S = R_pca(y_diag).fit(max_iter=500)
    plotContactMap(L, ofile = osp.join(dir, 'RPCA_L_diag.png'), vmax = np.max(y_diag))
    plotContactMap(S, ofile = osp.join(dir, 'RPCA_S_diag.png'), vmax = np.max(y_diag))




def scc_y_vs_y_rank1():
    dir = '/home/eric/sequences_to_contact_maps/dataset_01_16_22/samples/sample40'
    y = np.load(osp.join(dir, 'y.npy'))
    ycopy = y.copy()
    y_diag = np.load(osp.join(dir, 'y_diag.npy'))

    pca = PCA(n_components = 1)
    y_transform = pca.fit_transform(y)
    yhat = pca.inverse_transform(y_transform)
    y_transform = pca.fit_transform(y_diag)
    yhat_diag = pca.inverse_transform(y_transform)
    plotContactMap(yhat, ofile = osp.join(dir, 'yhat_rank1.png'), vmax = 'max')
    plotContactMap(yhat_diag, ofile = osp.join(dir, 'yhat_diag_rank1.png'), vmax = 'max')

    pca = PCA(n_components = 10)
    pca.fit(ycopy)
    plot_top_PCs(ycopy, 'name', dir)


    m, _ = y.shape
    triu_ind = np.triu_indices(m)

    overall_corr, _ = pearsonr(y[triu_ind], yhat[triu_ind])
    print('overall', overall_corr)
    scc_A, _ = pearsonr(y_diag[triu_ind], yhat_diag[triu_ind])
    print('scc_A', scc_A)


    corr_arr = np.zeros(m-2)
    corr_arr[0] = np.NaN
    scc_B = 0
    denom = np.sum([len(np.diagonal(y, offset = d)) for d in list(range(1, m-2))])
    weights = 0
    for d in range(1, m-2):
        # n-1, n, and 0 are NaN always, so skip
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, _ = pearsonr(y_diag, yhat_diag)
        corr_arr[d] = corr
        weight = len(y_diag) / denom
        weights += weight
        scc_B += weight * corr
    print(corr_arr)
    print('scc_B', scc_B)

    avg = np.nanmean(corr_arr)
    print('avg', avg)

    plt.plot(np.arange(m-2), corr_arr, color = 'black')
    plt.ylim(-0.5, 1)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(dir, 'distance_pearson_y_vs_y_rank_1.png'))
    plt.close()

def is_scc_weighted_mean():
    dir = '/home/eric/sequences_to_contact_maps/dataset_01_15_22/samples/sample40'
    replicate = '/home/eric/sequences_to_contact_maps/dataset_01_15_22/samples/sample40/PCA/k4/replicate1/iteration101'
    y = np.load(osp.join(dir, 'y.npy'))
    y_diag = np.load(osp.join(dir, 'y_diag.npy'))
    yhat = np.load(osp.join(replicate, 'y.npy'))
    yhat_diag = np.load(osp.join(replicate, 'y_diag.npy'))

    m, _ = y.shape
    triu_ind = np.triu_indices(m)

    overall_corr, _ = pearsonr(y[triu_ind], yhat[triu_ind])
    print(overall_corr)
    scc_A, _ = pearsonr(y_diag[triu_ind], yhat_diag[triu_ind])
    print(scc_A)


    corr_arr = np.zeros(m-2)
    corr_arr[0] = np.NaN
    scc_B = 0
    denom = np.sum([d for d in list(range(1, m-2))])
    for d in range(1, m-2):
        # n-1, n, and 0 are NaN always, so skip
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, _ = pearsonr(y_diag, yhat_diag)
        corr_arr[d] = corr
        weight = d / denom
        scc_B += weight * corr
    print(corr_arr)
    print(scc_B)

    avg = np.nanmean(corr_arr)
    print(avg)




if __name__ == '__main__':
    # main()
    # is_scc_weighted_mean()
    # scc_y_vs_y_rank1()
    test_robust_PCA()
    # check_seq()
    # find_mising_ids()
    # check_seq('dataset_11_03_21')
    # upper_traingularize_chis()
    # makeDirsForMaxEnt("dataset_08_29_21", 40)
