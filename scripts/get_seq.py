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
    seq_local = '../sequences_to_contact_maps'
    chip_seq_data_local = osp.join(seq_local, 'chip_seq_data')
    # "./project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--data_folder', type=str, default=osp.join(seq_local,'dataset_08_26_21'), help='location of input data')
    parser.add_argument('--sample', type=int, default=1201, help='sample id')
    parser.add_argument('--sample_folder', type=str, help='location of input data')
    parser.add_argument('--method', type=str, default='k_means', help='method for assigning particle types')
    parser.add_argument('--m', type=int, default=1024, help='number of particles (will crop contact map)')
    parser.add_argument('--p_switch', type=float, default=0.05, help='probability to switch bead assignment (for method = random)')
    parser.add_argument('--binarize', type=str2bool, default=False, help='true to binarize labels (not implemented for all methods)') # TODO
    parser.add_argument('--normalize', type=str2bool, default=False, help='true to normalize labels to [0,1] (not implemented for all methods)') # TODO
    parser.add_argument('--k', type=int, default=2, help='sequences to generate')
    parser.add_argument('--GNN_model_id', type=int, default=116, help='model id for ContactGNN')
    parser.add_argument('--epigenetic_data_folder', type=str, default=osp.join(chip_seq_data_local, 'fold_change_control/processed'), help='location of epigenetic data')
    parser.add_argument('--ChromHMM_data_file', type=str, default=osp.join(chip_seq_data_local, 'aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt'), help='location of ChromHMM data')
    parser.add_argument('--save_npy', action='store_true', help='true to save seq as .npy')
    parser.add_argument('--plot', action='store_true', help='true to plot seq as .png')


    args = parser.parse_args()
    args.clf = None
    args.X = None # X for silhouette_score
    args.method = args.method.lower()
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
    try:
        kmeans.fit(yKR)
    except ValueError as e:
        print(e)
        print('Not using KR')
        kmeans.fit(y)
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
    for i, (file, coverage) in enumerate(file_list[:k]):
        if coverage < min_coverage_prcnt / 100 * m:
            print("WARNING: mark {} has insufficient coverage: {}".format(file.split('_')[1], coverage))
        seq_i = np.load(osp.join(data_folder, file))
        seq_i = seq_i[start:end+1, 1]
        seq[:, i] = seq_i
    i += 1
    if i < k:
        print("Warning: insufficient data - only {} marks found".format(i))

    return seq


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
    return seq

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
    elif args.method == 'pca':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
        seq = get_PCA_seq(args.m, y_diag, args.k)
        format = '%.3e'
    elif args.method == 'pca_split':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
        seq = get_PCA_split_seq(args.m, y_diag, args.k)
        format = '%.3e'
    elif args.method == 'ground_truth':
        seq = np.load(osp.join(args.sample_folder, 'x.npy'))[:args.m, :]
        format = '%d'
    elif args.method == 'gnn':
        assert args.k == 2
        seq_path = "/home/erschultz/sequences_to_contact_maps/results/ContactGNN/{}/sample{}/z.npy".format(args.GNN_model_id, args.sample)
        if osp.exists(seq_path):
            seq = np.load(seq_path)[:args.m, :args.m]
        else:
            raise Exception('seq path does not exist: {}'.format(seq_path))
        format = '%.3e'
    elif args.method == 'k_means':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
        seq, args.clf = get_k_means_seq(args.m, y_diag, args.k)
        args.X = y_diag
        format = '%d'
    elif args.method == 'nmf':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
        seq, args.clf = get_nmf_seq(args.m, y_diag, args.k, args.binarize)
        args.X = y_diag
        format = '%.3e'
    elif args.method == 'epigenetic':
        seq = get_epigenetic_seq(args.epigenetic_data_folder, args.k)
        format = '%d'
    elif args.method == 'chromhmm':
        seq = get_ChromHMM_seq(args.ChromHMM_data_file, args.k)
        args.clf = None
        args.X = None
        format = '%d'
    else:
        raise Exception('Unkown method: {}'.format(args.method))

    m, k = seq.shape
    assert m == args.m, "m mismatch: seq has {} particles not {}".format(m, args.m)
    assert k == args.k, "k mismatch: seq has {} particle types not {}".format(k, args.k)
    writeSeq(seq, format, args.save_npy)

    if args.plot:
        if args.method == 'k_means' or (args.method == 'nmf' and args.binarize) or args.method == 'chromhmm':
            plot_seq_exclusive(seq, clf=args.clf, X=args.X)
        else:
            pass
            # TODO

def test():
    args = getArgs()
    args.k = 4
    y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
    # seq, args.clf = get_nmf_seq(args.m, y_diag, args.k, binarize = True)
    seq, args.clf = get_k_means_seq(args.m, y_diag, args.k, kr = True)
    args.X = y_diag
    format = '%.3e'

    plot_seq_exclusive(seq, clf=args.clf, X=args.X, show=True, save=False, title='test')

def test_epi():
    args = getArgs()
    args.k = 10
    seq = get_epigenetic_seq(args.epigenetic_data_folder, args.k)

def test_ChromHMM():
    args = getArgs()
    args.k = 9
    seq = get_ChromHMM_seq(args.ChromHMM_data_file, args.k)
    plot_seq_exclusive(seq, show=True, save=False, title='test')

if __name__ ==  "__main__":
    main()
    # test_epi()
    # test_ChromHMM()
