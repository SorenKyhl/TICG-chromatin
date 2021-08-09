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
    # "../../../../project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--data_folder', type=str, default='../../../../project2/depablo/erschultz/dataset_04_18_21', help='Location of input data')
    parser.add_argument('--sample', type=int, default=40, help='sample id')
    parser.add_argument('--method', type=str, default='random', help='method for assigning particle types')
    parser.add_argument('--m', type=int, default=1024, help='number of particles (will crop contact map)')
    parser.add_argument('--p_switch', type=float, default=0.05, help='probability to switch bead assignment')
    parser.add_argument('--k', type=int, default=2, help='sequences to generate')
    parser.add_argument('--GNN_model_id', type=int, default=116, help='model id for ContactGNN')


    args = parser.parse_args()
    if args.method != 'random':
        args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))
    return args

def get_random_seq(m, p_switch, k):
    seq = np.zeros((m, k))
    # vec[0, :] = np.random.choice([1,0], size = k)
    for j in range(k):
        for i in range(1, m):
            if seq[i-1, j] == 1:
                seq[i, j] = np.random.choice([1,0], p=[1 - p_switch, p_switch])
            else:
                seq[i, j] = np.random.choice([1,0], p=[p_switch, 1 - p_switch])

    return seq

def get_PCA_seq(m, y_diag, k):
    # pca = PCA()
    # pca.fit(y_diag)

    cov_arr = np.cov(y_diag)
    print(cov_arr)
    values, vectors = np.linalg.eig(cov_arr)
    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))

    print(np.sum(explained_variances[:5]), '\n', explained_variances[:5])

    seq = np.zeros((m, k))


    j = 0
    PC_count = k // 2 # 2 seqs per PC
    for pc_i in range(PC_count):
        # pc = pca.components_[pc_i]
        pc = vectors[:, pc_i]
        print(pc)
        plt.plot(pc, label = pc_i)

        pcpos = pc.copy()
        pcpos[pc < 0] = 0
        seq[:,j] = pcpos

        pcneg = pc.copy()
        pcneg[pc > 0] = 0
        seq[:,j+1] = pcneg * -1

        j += 2
    plt.legend()
    plt.show()

    return seq

def get_GNN_seq(args, y_diag):
    parser = getBaseParser()
    opt = parser.parse_args()
    opt.id = 116
    opt.model_type='ContactGNN'
    old_dir = os.getcwd()
    os.chdir('/../../sequences_to_contact_maps')
    opt = finalizeOpt(opt, parser)

    model = getModel(opt)
    model_name = osp.join(opt.ofile_folder, 'model.pt')
    if osp.exists(model_name):
        save_dict = torch.load(model_name, map_location=torch.device('cpu'))
        model.load_state_dict(save_dict['model_state_dict'])
        train_loss_arr = save_dict['train_loss']
        val_loss_arr = save_dict['val_loss']
        print('Model is loaded: {}'.format(model_name), file = opt.log_file)
    else:
        raise Exception('Model does not exist: {}'.format(model_name))
    model.eval()

    seq = model()

    os.chdir(old_dir)

def writeSeq(seq, format):
    print(seq)
    m, k = seq.shape
    for j in range(k):
        np.savetxt('seq{}.txt'.format(j), seq[:, j], fmt = format)

def main():
    args = getArgs()
    if args.method == 'random':
        seq = get_random_seq(args.m, args.p_switch, args.k)
        format = '%d'
    else:
        if args.method == 'PCA':
            y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
            seq = get_PCA_seq(args.m, y_diag, args.k)
            format = '%.3e'
        elif args.method == 'ground_truth':
            seq = np.load(osp.join(args.sample_folder, 'x.npy'))[:args.m, :]
            format = '%d'
        elif args.method == 'GNN':
            assert args.k == 2
            seq_path = '/../../sequences_to_contact_maps/results/ContactGNN/{}/sample{}/z.npy'.format(args.GNN_model_id, args.sample)
            if osp.exists(seq_path):
                seq = np.load(seq_path)[:args.m, :args.m]
            else:
                raise Exception('seq path does not exist: {}'.format(seq_path))
            format = '%.3e'
        elif args.method == 'k_means':
            y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
            kmeans = KMeans(n_clusters = args.k)
            kmeans.fit(y_diag)
            seq = np.zeros((args.m, args.k))
            seq[np.arange(args.m), kmeans.labels_] = 1
            format = '%d'

    for i in range(args.k):
        plt.plot(seq[:,i], label = i)
    plt.legend()
    plt.show()

    writeSeq(seq, format)


if __name__ ==  "__main__":
    main()
