import os
import os.path as osp
import sys

import numpy as np
import argparse

import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ensure that I can find knightRuiz
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from knightRuiz import knightRuiz

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    # '../../sequences_to_contact_maps/dataset_04_18_21'
    # "./project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--data_folder', type=str, default='../sequences_to_contact_maps/dataset_04_18_21', help='location of input data')
    parser.add_argument('--sample', type=int, default=2, help='sample id')
    parser.add_argument('--sample_folder', type=str, help='location of input data')
    parser.add_argument('--method', type=str, default='k_means', help='method for assigning particle types')
    parser.add_argument('--m', type=int, default=1024, help='number of particles (will crop contact map)')
    parser.add_argument('--p_switch', type=float, default=0.05, help='probability to switch bead assignment (for method = random)')
    parser.add_argument('--binarize', type=str2bool, default=False, help='true to binarize labels (not implemented for all methods)') # TODO
    parser.add_argument('--normalize', type=str2bool, default=False, help='true to normalize labels to [0,1] (not implemented for all methods)') # TODO
    parser.add_argument('--k', type=int, default=2, help='sequences to generate')
    parser.add_argument('--GNN_model_id', type=int, default=116, help='model id for ContactGNN')
    parser.add_argument('--save_npy', action='store_true', help='true to save seq as .npy')
    parser.add_argument('--plot', action='store_true', help='true to plot seq as .png')


    args = parser.parse_args()
    args.clf = None
    args.X = None # X for silhouette_score
    if args.method != 'random' and args.sample_folder is None:
        args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))
    return args

def str2bool(v):
    """
    Helper function for argparser, converts str to boolean for various string inputs.
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Inputs:
        v: string
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def get_k_means_seq(m, y, k, kr = True):
    if kr:
        yKR = np.log(knightRuiz(y))
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(yKR)
    seq = np.zeros((m, k))
    seq[np.arange(m), kmeans.labels_] = 1
    return seq, kmeans

def get_nmf_seq(m, y, k, binarize):
    nmf = NMF(n_components = k, max_iter = 1000)
    nmf.fit(y)
    H = nmf.components_

    print("NMF reconstruction error: {}".format(nmf.reconstruction_err_))

    if binarize:
        nmf.labels_ = np.argmax(H, axis = 0)
        seq = np.zeros((m, k))
        seq[np.arange(m), nmf.labels_] = 1
        return seq, nmf
    else:
        seq = H.T
        return seq, None

def writeSeq(seq, format, save_npy):
    m, k = seq.shape
    for j in range(k):
        np.savetxt('seq{}.txt'.format(j), seq[:, j], fmt = format)

    if save_npy:
        np.save('x.npy', seq)

def plot_seq_exclusive(seq, clf=None, X=None, show = False, save = True, title = None):
    '''Plotting function for mutually exclusive particle types'''
    # TODO make figure wider and less tall
    m, k = seq.shape
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = plt.cycler('color', cmap(ind))

    for i, c in enumerate(colors):
        x = np.argwhere(seq[:, i] == 1)
        plt.scatter(x, np.ones_like(x), label = i, color = c['color'], s=1)

    if X is not None and clf is not None:
        score = silhouette_score(X, clf.labels_)
        lower_title = '\nsilhouette score: {}'.format(np.round(score, 3))
    else:
        lower_title = ''

    plt.legend()
    ax = plt.gca()
    ax.axes.get_yaxis().set_visible(False)
    if title is not None:
        plt.title(title + lower_title, fontsize=16)
    if save:
        plt.savefig('seq.png')
    if show:
        plt.show()
    plt.close()

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
        seq, args.clf = get_k_means_seq(args.m, y_diag, args.k)
        args.X = y_diag
        format = '%d'
    elif args.method == 'nmf':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        seq, args.clf = get_nmf_seq(args.m, y_diag, args.k, args.binarize)
        args.X = y_diag
        format = '%.3e'
    else:
        raise Exception('Unkown method: {}'.format(args.method))

    m, k = seq.shape
    assert m == args.m, "m mismatch: seq has {} particles not {}".format(m, args.m)
    assert k == args.k, "k mismatch: seq has {} particle types not {}".format(k, args.k)
    writeSeq(seq, format, args.save_npy)

    if args.plot:
        if args.method == 'k_means' or (args.method == 'nmf' and args.binarize):
            plot_seq_exclusive(seq, clf=args.clf, X=args.X)
        else:
            pass
            # TODO

def test():
    args = getArgs()
    y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
    seq, args.clf = get_nmf_seq(args.m, y_diag, args.k, binarize = True)
    # seq, args.clf = get_k_means_seq(args.m, y_diag, args.k, kr = True)
    args.X = y_diag
    format = '%.3e'

    plot_seq_exclusive(seq, clf=args.clf, X=args.X, show=True, save=True, title='help')


if __name__ ==  "__main__":
    main()
