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

# ensure that I can find knightRuiz and get_config
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from knightRuiz import knightRuiz
from get_config import LETTERS, str2bool, str2int

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from plotting_functions import plotContactMap

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    seq_local = '../sequences_to_contact_maps'
    chip_seq_data_local = osp.join(seq_local, 'chip_seq_data')
    # "./project2/depablo/erschultz/dataset_04_18_21"

    # input data args
    parser.add_argument('--data_folder', type=str, default=osp.join(seq_local,'dataset_08_26_21'), help='location of input data')
    parser.add_argument('--sample', type=int, default=40, help='sample id')
    parser.add_argument('--sample_folder', type=str, help='location of input data')

    # standard args
    parser.add_argument('--method', type=str, default='k_means', help='method for assigning particle types')
    parser.add_argument('--m', type=int, default=1024, help='number of particles (will crop contact map)')
    parser.add_argument('--k', type=str2int, default=2, help='sequences to generate')

    # args for specific methods
    parser.add_argument('--seed', type=int, help='random seed for numpy')
    parser.add_argument('--model_path', type=str, help='path to GNN model')
    parser.add_argument('--use_energy', type=str2bool, default=False)
    parser.add_argument('--correct_energy', type=str2bool, default=False, help='True to correct S by dividing all non-diagonal entries by 2')
    parser.add_argument('--epigenetic_data_folder', type=str, default=osp.join(chip_seq_data_local, 'fold_change_control/processed'), help='location of epigenetic data')
    parser.add_argument('--ChromHMM_data_file', type=str, default=osp.join(chip_seq_data_local, 'aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt'), help='location of ChromHMM data')
    parser.add_argument('--p_switch', type=float, default=0.05, help='probability to switch bead assignment (for method = random)')

    # post-processing args
    parser.add_argument('--binarize', type=str2bool, default=False, help='true to binarize labels (not implemented for all methods)') # TODO
    parser.add_argument('--normalize', type=str2bool, default=False, help='true to normalize labels to [0,1] (or [-1, 1] for some methods) (not implemented for all methods)') # TODO
    parser.add_argument('--save_npy', action='store_true', help='true to save seq as .npy')
    parser.add_argument('--plot', action='store_true', help='true to plot seq as .png')
    parser.add_argument('--relabel', type=str2None, help='specify mark combinations to be relabled (e.g. AB-C will relabel AB mark pairs as mark C)')

    args = parser.parse_args()
    args.labels = None
    args.X = None # X for silhouette_score
    args.method = args.method.lower()
    args.dataset = osp.split(args.data_folder)[1]

    if args.method != 'random' and args.sample_folder is None:
        args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))

    # check params
    if args.relabel is not None:
        assert args.method == 'random', 'relabel currently only supported for random'
    if args.binarize:
        if args.method in {'k_means', 'chromhmm'}:
            print("{} is binarized by default".format(args.method))
        elif args.method in {'nmf'}:
            pass
        else:
            raise Exception('binarize not yet supported for {}'.format(args.method))
    if args.normalize:
        if args.method in {}:
            pass
        else:
            raise Exception('normalize not yet supported for {}'.format(args.method))

    return args

def str2None(v):
    """
    Helper function for argparser, converts str to None if str == 'none'

    Returns the string otherwise.

    Inputs:
        v: string
    """
    if v is None:
        return v
    elif isinstance(v, str):
        if v.lower() == 'none':
            return None
        else: return v
    else:
        raise argparse.ArgumentTypeError('String value expected.')

def get_random_seq(m, p_switch, k, relabel, seed):
    rng = np.random.default_rng(seed)
    if relabel is not None:
        k -= 1
    seq = np.zeros((m, k))
    seq[0, :] = np.random.choice([1,0], size = k)
    for j in range(k):
        for i in range(1, m):
            if seq[i-1, j] == 1:
                seq[i, j] = rng.choice([1,0], p=[1 - p_switch, p_switch])
            else:
                seq[i, j] = rng.choice([1,0], p=[p_switch, 1 - p_switch])

    if relabel is not None:
        seq = relabel_seq(seq, relabel)

    return seq

def relabel_seq(seq, relabel_str):
    '''
    Relabels seq according to relabel_str.

    Inputs:
        seq: m x k np array
        relabel_str: string of format <old>-<new>

    Example:
    consider: <old> = AB, <new> = D, seq is m x 3
    Any particle with both label A and label B, will be relabeled to have
    label C and neither A nor B. Label C will be unaffected.

    Note that LETTERS.find(new) must be >= k
    (i.e label <new> cannot be present in seq already)
    '''
    m, k = seq.shape

    old, new = relabel_str.split('-')
    new_label = LETTERS.find(new)
    assert new_label >= k # new label cannot already be present
    old_labels = [LETTERS.find(i) for i in old]
    all_labels = [LETTERS.find(i) for i in old+new]

    new_seq = np.zeros((m, k+1))
    new_seq[:, :k] = seq

    # find where to assing new_label
    where = np.ones(m) # all True
    for i in old_labels:
        where_i = seq[:, i] == 1
        where = np.logical_and(where, seq[:, i] == 1)

    # assign new_label
    new_seq[:, new_label] = where
    # delete old_labels
    for i in old_labels:
        new_seq[:, i] -= where

    # check that new_label is mutually exclusive from old_labels
    row_sum = np.sum(new_seq[:, all_labels], axis = 1)
    assert np.all(row_sum <= 1)

    return new_seq

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

def get_PCA_seq(input, k, normalize):
    '''
    Defines seq based on PCs of input.

    Inputs:
        input: matrix to perform PCA on
        k: numper of particle types / principal components to use
        normalize: True to normalize particle types / principal components to [-1, 1]

    Outputs:
        seq: array of particle types
    '''
    m, _ = input.shape
    pca = PCA()
    pca.fit(input)
    seq = np.zeros((m, k))

    for j in range(k):
        pc = pca.components_[j]

        if normalize:
            min = np.min(pc)
            max = np.max(pc)
            if max > abs(min):
                val = max
            else:
                val = abs(min)

            # multiply by scale such that val x scale = 1
            scale = 1/val
            pc *= scale

        seq[:,j] = pc

    return seq

def get_k_means_seq(m, y, k, kr = True):
    if kr:
        yKR = np.log(knightRuiz(y))
    kmeans = KMeans(n_clusters = k)
    try:
        kmeans.fit(yKR)
    except ValueError as e:
        print(e)
        print('Not using KR')
        kmeans.fit(y)
    seq = np.zeros((m, k))
    seq[np.arange(m), kmeans.labels_] = 1
    return seq, kmeans.labels_

def get_nmf_seq(m, y, k, binarize):
    nmf = NMF(n_components = k, max_iter = 1000)
    nmf.fit(y)
    H = nmf.components_

    print("NMF reconstruction error: {}".format(nmf.reconstruction_err_))

    if binarize:
        nmf.labels_ = np.argmax(H, axis = 0)
        seq = np.zeros((m, k))
        seq[np.arange(m), nmf.labels_] = 1
        return seq, nmf.labels_
    else:
        seq = H.T
        return seq, None

def get_epigenetic_seq(data_folder, k, start=35000000, end=60575000, res=25000, chr='2', min_coverage_prcnt=5):
    '''
    Loads experimental epigenetic data from data_folder to use as particle types.

    Inputs:
        data_folder: location of epigenetic data - file format: <chr>_*.npy
        k: number of particle types to use - will pick top k data files from data_folder with most coverage
        start: start in base pairs
        end: end location in base pairs
        res: resolution of data/simulation
        chr: chromosome
        min_coverage_prcnt: minimum percent of particle of given particle type

    Outputs:
        seq: particle type np array of shape m x k
    '''
    start = int(start / res)
    end = int(end / res)
    m = end - start + 1 # number of particle in simulation

    # store file names and coverage in list
    file_list = [] # list of tuples (file_name, coverage)
    for file in os.listdir(data_folder):
        if file.startswith(chr + "_"):
            seq_i = np.load(osp.join(data_folder, file))
            seq_i = seq_i[start:end+1, 1]
            coverage = np.sum(seq_i)
            file_list.append((file, coverage))

    # sort based on coverage
    file_list = sorted(file_list, key = lambda pair: pair[1], reverse = True)
    print(file_list[:k], )

    # choose k marks with most coverage
    seq = np.zeros((m, k))
    marks = []
    for i, (file, coverage) in enumerate(file_list[:k]):
        mark = file.split('_')[1]
        marks.append(mark)
        if coverage < min_coverage_prcnt / 100 * m:
            print("WARNING: mark {} has insufficient coverage: {}".format(mark, coverage))
        seq_i = np.load(osp.join(data_folder, file))
        seq_i = seq_i[start:end+1, 1]
        seq[:, i] = seq_i
    i += 1
    if i < k:
        print("Warning: insufficient data - only {} marks found".format(i))

    return seq, marks

def get_ChromHMM_seq(ifile, k, start=35000000, end=60575000, res=25000, min_coverage_prcnt=5):
    start = int(start / res)
    end = int(end / res)
    m = end - start + 1 # number of particle in simulation

    with open(ifile, 'r') as f:
        f.readline()
        f.readline()
        states = np.array([int(state.strip()) - 1 for state in f.readlines()])
        # subtract 1 to convert to 0 based indexing
        states = states[start:end+1]


    seq = np.zeros((m, 15))
    seq[np.arange(m), states] = 1

    coverage_arr = np.sum(seq, axis = 0) # number of beads of each particle type

    # sort based on coverage
    insufficient_coverage = np.argwhere(coverage_arr < min_coverage_prcnt * m / 100).flatten()

    # exclude marks with no coverage
    for mark in insufficient_coverage:
        print("Mark {} has insufficient coverage: {}".format(mark, coverage_arr[mark]))
    seq = np.delete(seq, insufficient_coverage, 1)

    assert seq.shape[1] == k

    # get labels
    labels = np.where(seq == np.ones((m, 1)))[1]
    if len(labels) == m:
        # this is only true if all marks have sufficient coverage
        # i.e. none were deleted
        pass
    else:
        labels = None

    return seq, labels

def get_energy_gnn(model_path, sample):
    '''
    Loads output from GNN model to use as pairwise energy matrix, S

    Inputs:
        model_path: path to model results
        sample: sample id (int)

    Outputs:
        s: np array of pairwise energies
    '''
    # determine model_type
    model_type = osp.split(osp.split(model_path)[0])[1]
    print(model_type)

    assert model_type == 'ContactGNNEnergy', "Unrecognized model_type: {}".format(model_type)
    s_path = osp.join(model_path, "sample{}/energy_hat.txt".format(sample))
    if osp.exists(s_path):
        s = np.loadtxt(s_path)
    else:
        raise Exception('s_path does not exist: {}'.format(s_path))

    return s

def get_seq_gnn(k, model_path, sample, normalize):
    '''
    Loads output from GNN model to use as particle types, seq

    Inputs:
        k: number of particle types
        model_path: path to model results
        sample: sample id (int)
        normalize: True to normalize seq to [-1,1] (only for ContactGNNEnergy)

    Outputs:
        seq: particle types
    '''
    model_type = osp.split(osp.split(model_path)[0])[1]
    print(model_type)

    if model_type == 'ContactGNN':
        z_path = osp.join(model_path, "sample{}/z.npy".format(sample))
        if osp.exists(z_path):
            seq = np.load(z_path)
            assert seq.shape[1] == k
        else:
            raise Exception('z_path does not exist: {}'.format(z_path))
    elif model_type == 'ContactGNNEnergy':
        s_path = osp.join(model_path, "sample{}/energy_hat.txt".format(sample))
        if osp.exists(s_path):
            s = np.loadtxt(s_path)
        else:
            raise Exception('s_path does not exist: {}'.format(s_path))

        seq = get_PCA_seq(s, k, normalize)
    else:
        raise Exception("Unrecognized model_type: {}".format(model_type))

    return seq

def writeSeq(seq, format, save_npy):
    m, k = seq.shape
    for j in range(k):
        np.savetxt('seq{}.txt'.format(j), seq[:, j], fmt = format)

    if save_npy:
        np.save('x.npy', seq)

def plot_seq_exclusive(seq, labels=None, X=None, show = False, save = True, title = None):
    '''Plotting function for mutually exclusive binary particle types'''
    # TODO make figure wider and less tall
    m, k = seq.shape
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = plt.cycler('color', cmap(ind))

    for i, c in enumerate(colors):
        x = np.argwhere(seq[:, i] == 1)
        plt.scatter(x, np.ones_like(x), label = i, color = c['color'], s=1)

    if X is not None and labels is not None:
        score = silhouette_score(X, labels)
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

def plot_seq_binary(seq, show = False, save = True, title = None, labels = None, x_axis = True):
    '''Plotting function for non mutually exclusive binary particle types'''
    # TODO make figure wider and less tall
    m, k = seq.shape
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = plt.cycler('color', cmap(ind))

    for i, c in enumerate(colors):
        x = np.argwhere(seq[:, i] == 1)
        if labels is None:
            label_i = i
        else:
            label_i = labels[i]
        plt.scatter(x, np.ones_like(x) * i, label = label_i, color = c['color'], s=1)

    plt.legend()
    ax = plt.gca()
    ax.axes.get_yaxis().set_visible(False)
    if not x_axis:
        ax.axes.get_xaxis().set_visible(False)
    if title is not None:
        plt.title(title, fontsize=16)
    if save:
        plt.savefig('seq.png')
    if show:
        plt.show()
    plt.close()

def main():
    args = getArgs()
    if args.method == 'random':
        seq = get_random_seq(args.m, args.p_switch, args.k, args.relabel, args.seed)
        format = '%d'
    elif args.method == 'pca':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
        seq = get_PCA_seq(y_diag, args.k, args.normalize)
        format = '%.3e'
    elif args.method == 'pca_split':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
        seq = get_PCA_split_seq(args.m, y_diag, args.k)
        format = '%.3e'
    elif args.method == 'ground_truth':
        seq = np.load(osp.join(args.sample_folder, 'x.npy'))[:args.m, :]
        if args.use_energy:
            chi = np.load(osp.join(args.sample_folder, 'chis.npy'))[:args.m, :]
            s = seq @ chi @ seq.T
            format = '%.3e'
        else:
            format = '%d'

    elif args.method == 'k_means':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
        seq, args.labels = get_k_means_seq(args.m, y_diag, args.k)
        args.X = y_diag
        format = '%d'
    elif args.method == 'nmf':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
        seq, args.labels = get_nmf_seq(args.m, y_diag, args.k, args.binarize)
        args.X = y_diag
        format = '%.3e'
    elif args.method == 'epigenetic':
        seq, marks = get_epigenetic_seq(args.epigenetic_data_folder, args.k)
        format = '%d'
    elif args.method == 'chromhmm':
        seq, labels = get_ChromHMM_seq(args.ChromHMM_data_file, args.k)
        format = '%d'
    elif args.method == 'gnn':
        argparse_path = osp.join(args.model_path, 'argparse.txt')
        with open(argparse_path, 'r') as f:
            for line in f:
                if line == '--data_folder\n':
                    break
            data_folder = f.readline().strip()
            dataset = osp.split(data_folder)[1]
        assert dataset == args.dataset, 'Dataset mismatch: {} vs {}'.format(dataset, args.dataset)

        if args.use_energy:
            s = get_energy_gnn(args.model_path, args.sample)
        else:
            seq = get_seq_gnn(args.k, args.model_path, args.sample, args.normalize)
        format = '%.3e'
    else:
        raise Exception('Unkown method: {}'.format(args.method))

    if args.use_energy:
        m1, m2 = s.shape
        assert m1 == m2, "shape mismatch, {} vs {}".format(m1, m2)
        assert m1 == args.m
        if args.correct_energy:
            diag = np.diagonal(s).copy()
            s /= 2
            np.fill_diagonal(s, diag)
        print(s)
        np.savetxt('s_matrix.txt', s, fmt = format)
    else:
        m, k = seq.shape
        np.set_printoptions(threshold=100)
        for i in range(k):
            print(repr(seq[:100, i]))
        assert m == args.m, "m mismatch: seq has {} particles not {}".format(m, args.m)
        assert k == args.k, "k mismatch: seq has {} particle types not {}".format(k, args.k)
        writeSeq(seq, format, args.save_npy)

    if args.plot:
        if args.use_energy:
            plotContactMap(s, 's_matrix.png', vmin = 'min', vmax = 'max', cmap = 'blue-red')
        elif args.method in {'k_means', 'chromhmm'} or (args.method == 'nmf' and args.binarize):
            plot_seq_exclusive(seq, labels=args.labels, X=args.X)
        elif args.binarize:
            plot_seq_binary(seq)


def test_nmf_k_means():
    args = getArgs()
    args.k = 4
    y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
    args.X = y_diag

    seq, args.labels = get_nmf_seq(args.m, y_diag, args.k, binarize = True)
    plot_seq_exclusive(seq, labels=args.labels, X=args.X, show=True, save=False, title='nmf-binarize test')

    seq, args.labels = get_nmf_seq(args.m, y_diag, args.k, binarize = False)

    seq, args.labels = get_k_means_seq(args.m, y_diag, args.k, kr = True)
    plot_seq_exclusive(seq, labels=args.labels, X=args.X, show=True, save=False, title='k_means test')

def test_random():
    args = getArgs()
    args.k = 4
    args.m = 1000
    args.relabel = 'AB-D'
    seq = get_random_seq(args.m, args.p_switch, args.k, args.relabel, args.seed)

    plot_seq_binary(seq, show = True, save = False, title = 'test')

def test_epi():
    args = getArgs()
    args.k = 6
    seq, marks = get_epigenetic_seq(args.epigenetic_data_folder, args.k)
    print(marks)
    plot_seq_binary(seq, show = True, save = False, title = None, labels = marks, x_axis = False)

def test_ChromHMM():
    args = getArgs()
    args.k = 15
    y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
    args.X = y_diag
    seq, args.labels = get_ChromHMM_seq(args.ChromHMM_data_file, args.k, min_coverage_prcnt = 0)
    plot_seq_exclusive(seq, labels=args.labels, X=args.X, show=True, save=False, title='ChromHMM test')

def test_GNN():
    args = getArgs()
    args.k = 2
    args.model_path = '../sequences_to_contact_maps/results/ContactGNNEnergy/26'
    args.normalize = True

    seq = get_seq_gnn(args.k, args.model_path, args.sample, args.normalize, args.use_energy)


if __name__ ==  "__main__":
    main()
    # test_nmf_k_means()
    # test_random()
    # test_epi()
    # test_ChromHMM()
    # test_GNN()
