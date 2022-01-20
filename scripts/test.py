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
    dir = '/home/eric/dataset_test/samples'
    A = 'sample82'
    B = 'sample83'
    dirA=osp.join(dir, A)
    dirB=osp.join(dir, B)

    # B and D has diag on
    # C and D has diag normalized
    xA = np.load(osp.join(dirA, 'x.npy'))
    xB = np.load(osp.join(dirB, 'x.npy'))

    sA = np.load(osp.join(dirA, 's.npy'))
    sB = np.load(osp.join(dirB, 's.npy'))

    yA = np.load(osp.join(dirA, 'y.npy'))
    yB = np.load(osp.join(dirB, 'y.npy'))
    yC = np.load(osp.join(dirA, 'y_diag.npy'))
    yD = np.load(osp.join(dirB, 'y_diag.npy'))

    # Compare PCs ##
    print("\nA")
    PC_yA = plot_top_PCs(yA, 'yA', odir = dirA, plot = True)

    print("\n\nB")
    PC_yB = plot_top_PCs(yB, 'yB', odir = dirB, plot = True)
    stat = pearsonround(PC_yA[0], PC_yB[0])
    print("Correlation between PC 1 of A and B: ", stat)

    print("\nC")
    PC_yC = plot_top_PCs(yC, 'ydiagC', odir = dirA, plot = True)
    stat = pearsonround(PC_yA[0], PC_yC[0])
    print("Correlation between PC 1 of A and C: ", stat)
    stat = pearsonround(PC_yA[1], PC_yC[1])
    print("Correlation between PC 2 of A and C: ", stat)

    print("\nD")
    PC_yD = plot_top_PCs(yD, 'ydiagD', odir = dirB, plot = True)
    stat = pearsonround(PC_yD[0], PC_yC[0])
    print("Correlation between PC 1 of C and D: ", stat)
    stat = pearsonround(PC_yD[1], PC_yC[1])
    print("Correlation between PC 2 of C and D: ", stat)

    # print("\nS")
    # print(f'Rank: {np.linalg.matrix_rank(sA)}')
    # PC_s = plot_top_PCs(sA)
    # stat = pearsonround(PC_yC[0], PC_s[0])
    # print("Correlation between PC 1 of C and S: ", stat)
    # stat = pearsonround(PC_yD[0], PC_s[0])
    # print("Correlation between PC 1 of D and S: ", stat)

    s_sym = (sA + sA.T)/2
    print("\n\nS_sym")
    print(f'Rank: {np.linalg.matrix_rank(s_sym)}')
    PC_s_sym = plot_top_PCs(s_sym, 's_sym', odir = dirA, plot = True)
    stat = pearsonround(PC_yC[0], PC_s_sym[0])
    print("Correlation between PC 1 of C and S_sym: ", stat)
    stat = pearsonround(PC_yC[1], PC_s_sym[1])
    print("Correlation between PC 2 of C and S_sym: ", stat)
    stat = pearsonround(PC_yD[0], PC_s_sym[0])
    print("Correlation between PC 1 of D and S_sym: ", stat)
    stat = pearsonround(PC_yD[1], PC_s_sym[1])
    print("Correlation between PC 2 of D and S_sym: ", stat)

def main2():
    dir = '/home/eric/sequences_to_contact_maps/dataset_09_21_21/samples/sample1'
    y = np.load(osp.join(dir, 'y.npy'))
    np.savetxt(osp.join(dir, 'y.txt'), y)


if __name__ == '__main__':
    # main2()
    check_seq()
    # find_mising_ids()
    # check_seq('dataset_11_03_21')
    # upper_traingularize_chis()
    # makeDirsForMaxEnt("dataset_08_29_21", 40)
