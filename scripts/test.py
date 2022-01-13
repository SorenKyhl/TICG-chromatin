import sys
import os
import os.path as osp

import numpy as np

# ensure that I can find contact_map
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from makeLatexTable import METHODS

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.dataset_classes import make_dataset
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
    dir = "/project2/depablo/erschultz/dataset_12_12_21/samples"
    # dir = "/home/eric/sequences_to_contact_maps/dataset_11_03_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            file_dir = osp.join(dir, file)
            x_linear_file = osp.join(file_dir, 'x_linear.npy')
            xfile = osp.join(file_dir, 'x.npy')
            x = np.load(xfile)
            m, k = x.shape
            seq = np.zeros((m ,k))
            for i in range(k):
                seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                seq[:, i] = seq_i
            if not np.array_equal(seq, x):
                ids_to_check.add(int(file[6:]))
                print(int(file[6:]))

            if osp.exists(x_linear_file):
                x_linear = np.load(x_linear_file)
                np.save(osp.join(file_dir, 'psi.npy'), x_linear)

def check_seq(dataset):
    # dir = "/project2/depablo/erschultz"
    dir = '/home/eric/sequences_to_contact_maps'
    if dataset is None:
        datasets = os.listdir(dir)
    else:
        datasets = [dataset]
    print(datasets)
    for dataset in datasets:
        if dataset.startswith("dataset") and osp.isdir(osp.join(dir, dataset)):
            ids_to_check = set()
            print(dataset)
            dataset_samples = osp.join(dir, dataset, 'samples')
            for file in os.listdir(dataset_samples):
                passed = True
                if file.startswith('sample'):
                    file_dir = osp.join(dataset_samples, file)
                    x = np.load(osp.join(file_dir, 'x.npy'))
                    m, k = x.shape
                    seq = np.zeros((m ,k))
                    for i in range(k):
                        seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                        seq[:, i] = seq_i
                    if not np.array_equal(seq, x):
                        ids_to_check.add(int(file[6:]))
                        print('fail1')
                        passed = False
                        # np.save(osp.join(file_dir, 'x.npy'), seq)

                    x_linear = np.load(osp.join(file_dir, 'x_linear.npy'))
                    m, k = x_linear.shape
                    seq = np.zeros((m ,k))
                    for i in range(k):
                        seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                        seq[:, i] = seq_i
                    if not np.array_equal(seq, x_linear):
                        ids_to_check.add(int(file[6:]))
                        print('fail2')
                        passed = False

                    if dataset.startswith("dataset_11_03"):
                    # if int(file[6:]) > 10:
                        row_sum = np.sum(x_linear[:, [0,1,3]], axis = 1)
                        if not np.all(row_sum <= 1):
                            ids_to_check.add(int(file[6:]))
                            passed = False
                            print('fail3')

                    print(f'{file} passed: {passed}')

            print(sorted(ids_to_check))

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
    dir = '/home/eric/dataset_test/samples'
    # 81 has diag on
    x80 = np.load(osp.join(dir, 'sample80', 'x.npy'))
    x81 = np.load(osp.join(dir, 'sample81', 'x.npy'))

    s80 = np.load(osp.join(dir, 'sample80', 's.npy'))
    s81 = np.load(osp.join(dir, 'sample81', 's.npy'))

    y80 = np.load(osp.join(dir, 'sample80', 'y.npy'))
    y81 = np.load(osp.join(dir, 'sample81', 'y.npy'))
    ydiag80 = np.load(osp.join(dir, 'sample80', 'y_diag.npy'))
    ydiag81 = np.load(osp.join(dir, 'sample81', 'y_diag.npy'))

    # Compare PCs ##
    print("\nY80")
    PC_y80 = plot_top_PCs(y80)

    print("\nY_diag80")
    PC_ydiag80 = plot_top_PCs(ydiag80)
    stat = pearsonround(PC_y80[0], PC_ydiag80[0])
    print("Correlation between PC 1 of y_diag80 and y80: ", stat)
    stat = pearsonround(PC_y80[1], PC_ydiag80[1])
    print("Correlation between PC 2 of y_diag80 and y80: ", stat)


    print("\nY_diag81")
    PC_ydiag81 = plot_top_PCs(ydiag81)
    stat = pearsonround(PC_ydiag81[0], PC_ydiag80[0])
    print("Correlation between PC 1 of y_diag80 and y_diag81: ", stat)
    stat = pearsonround(PC_ydiag81[1], PC_ydiag80[1])
    print("Correlation between PC 2 of y_diag80 and y_diag81: ", stat)

    print("\nS")
    print(f'Rank: {np.linalg.matrix_rank(s80)}')
    PC_s = plot_top_PCs(s80)
    stat = pearsonround(PC_ydiag80[0], PC_s[0])
    print("Correlation between PC 1 of y_diag80 and S: ", stat)
    stat = pearsonround(PC_ydiag81[0], PC_s[0])
    print("Correlation between PC 1 of y_diag81 and S: ", stat)

    s_sym = (s80 + s80.T)/2
    print("\nS_sym")
    print(f'Rank: {np.linalg.matrix_rank(s_sym)}')
    PC_s_sym = plot_top_PCs(s_sym)
    stat = pearsonround(PC_ydiag80[0], PC_s_sym[0])
    print("Correlation between PC 1 of y_diag80 and S_sym: ", stat)
    stat = pearsonround(PC_ydiag80[1], PC_s_sym[1])
    print("Correlation between PC 2 of y_diag80 and S_sym: ", stat)
    stat = pearsonround(PC_ydiag81[0], PC_s_sym[0])
    print("Correlation between PC 1 of y_diag81 and S_sym: ", stat)
    stat = pearsonround(PC_ydiag81[1], PC_s_sym[1])
    print("Correlation between PC 2 of y_diag81 and S_sym: ", stat)


if __name__ == '__main__':
    main()
    # write_psi()
    # find_mising_ids()
    # check_seq('dataset_11_03_21')
    # upper_traingularize_chis()
    # makeDirsForMaxEnt("dataset_08_29_21", 40)
