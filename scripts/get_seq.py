import os
import os.path as osp
import sys

import numpy as np
import argparse

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    # '../../sequences_to_contact_maps/dataset_04_18_21'
    # "./project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--data_folder', type=str, default='../sequences_to_contact_maps/dataset_04_18_21', help='location of input data')
    parser.add_argument('--sample', type=int, default=2, help='sample id')
    parser.add_argument('--sample_folder', type=str, help='location of input data')
    parser.add_argument('--method', type=str, default='PCA', help='method for assigning particle types')
    parser.add_argument('--m', type=int, default=1024, help='number of particles (will crop contact map)')
    parser.add_argument('--p_switch', type=float, default=0.05, help='probability to switch bead assignment')
    parser.add_argument('--k', type=int, default=2, help='sequences to generate')
    parser.add_argument('--GNN_model_id', type=int, default=116, help='model id for ContactGNN')
    parser.add_argument('--save_npy', action='store_true', help='true to save seq as .npy')


    args = parser.parse_args()
    if args.method != 'random' and args.sample_folder is None:
        args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))
    return args

def get_random_seq(m, p_switch, k):
    seq = np.zeros((m, k))
    seq[0, :] = np.random.choice([1,0], size = k)
    for j in range(k):
        for i in range(1, m):
            if seq[i-1, j] == 1:
                seq[i, j] = np.random.choice([1,0], p=[1 - p_switch, p_switch])
            else:
                seq[i, j] = np.random.choice([1,0], p=[p_switch, 1 - p_switch])

    return seq

def get_PCA_split_seq(m, y_diag, k):
    pca = PCA()
    pca.fit(y_diag)
    seq = np.zeros((m, k))

    j = 0
    PC_count = k // 2 # 2 seqs per PC
    for pc_i in range(PC_count):
        pc = pca.components_[pc_i]

        pcpos = pc.copy()
        pcpos[pc < 0] = 0 # zero negative part
        seq[:,j] = pcpos

        pcneg = pc.copy()
        pcneg[pc > 0] = 0 # zero positive part
        seq[:,j+1] = pcneg * -1
        j += 2
    return seq

def get_PCA_seq(m, y_diag, k):
    pca = PCA()
    pca.fit(y_diag)
    seq = np.zeros((m, k))

    for j in range(k):
        seq[:,j] = pca.components_[j]

    return seq

def get_k_means_seq(m, y_diag, k):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(y_diag)
    seq = np.zeros((m, k))
    seq[np.arange(m), kmeans.labels_] = 1
    return seq

def writeSeq(seq, format, save_npy):
    m, k = seq.shape
    for j in range(k):
        np.savetxt('seq{}.txt'.format(j), seq[:, j], fmt = format)

    if save_npy:
        np.save('x.npy', seq)

def main():
    args = getArgs()
    if args.method == 'random':
        seq = get_random_seq(args.m, args.p_switch, args.k)
        format = '%d'
    elif args.method == 'PCA':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        seq = get_PCA_seq(args.m, y_diag, args.k)
        format = '%.3e'
    elif args.method == 'PCA_split':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        seq = get_PCA_split_seq(args.m, y_diag, args.k)
        format = '%.3e'
    elif args.method == 'ground_truth':
        seq = np.load(osp.join(args.sample_folder, 'x.npy'))[:args.m, :]
        format = '%d'
    elif args.method == 'GNN':
        assert args.k == 2
        seq_path = "/home/erschultz/sequences_to_contact_maps/results/ContactGNN/{}/sample{}/z.npy".format(args.GNN_model_id, args.sample)
        if osp.exists(seq_path):
            seq = np.load(seq_path)[:args.m, :args.m]
        else:
            raise Exception('seq path does not exist: {}'.format(seq_path))
        format = '%.3e'
    elif args.method == 'k_means':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        seq = get_k_means_seq(args.m, y_diag, args.k)
        format = '%d'
    else:
        raise Exception('Unkown method: {}'.format(args.method))

    for i in range(args.k):
        plt.plot(seq[:,i], label = i)
    plt.legend()
    plt.show()

    writeSeq(seq, format, args.save_npy)


if __name__ ==  "__main__":
    main()
