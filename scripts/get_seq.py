import os
import os.path as osp
import sys

import numpy as np
import argparse
import re

import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, NMF, KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ensure that I can find knightRuiz
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from knightRuiz import knightRuiz

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from plotting_functions import plotContactMap
from neural_net_utils.argparseSetup import str2bool, str2int, str2None
from neural_net_utils.utils import calculate_E_S

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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
    parser.add_argument('--exclusive', type=str2bool, default=False, help='True to use mutually exusive label (for random method)')
    parser.add_argument('--model_path', type=str, help='path to GNN model')
    parser.add_argument('--epigenetic_data_folder', type=str, default=osp.join(chip_seq_data_local, 'fold_change_control/processed'), help='location of epigenetic data')
    parser.add_argument('--ChromHMM_data_file', type=str, default=osp.join(chip_seq_data_local, 'aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt'),
                        help='location of ChromHMM data')
    parser.add_argument('--p_switch', type=float, default=0.05, help='probability to switch bead assignment (for method = random)')
    parser.add_argument('--kernel', type=str, default='poly', help='kernel for kernel PCA')

    # post-processing args
    parser.add_argument('--save_npy', action='store_true', help='true to save seq as .npy')
    parser.add_argument('--plot', action='store_true', help='true to plot seq as .png')


    args = parser.parse_args()
    args.input = None # input in {y, x, psi}
    args.binarize = False # True to binarize labels (not implemented for all methods)') # TODO
    args.normalize = False # True to normalize labels to [0,1] (or [-1, 1] for some methods) (not implemented for all methods)') # TODO
    args.use_ematrix = False
    args.use_smatrix = False
    args.append_random = False # True to append random seq
    args.method = args.method.lower()
    process_method(args)

    args.labels = None
    args.X = None # X for silhouette_score
    args.dataset = osp.split(args.data_folder)[1]
    if args.method != 'random' and args.sample_folder is None:
        args.sample_folder = osp.join(args.data_folder, 'samples', f'sample{args.sample}')

    # check params
    if args.binarize:
        if args.method in {'k_means', 'chromhmm'}:
            print(f"{args.method} is binarized by default")
        elif args.method in {'nmf'}:
            pass
        else:
            raise Exception(f'binarize not yet supported for {args.method}')
    if args.normalize:
        if args.method in {}:
            pass
        else:
            raise Exception(f'normalize not yet supported for {args.method}')
    if args.exclusive:
        assert args.method == 'random', 'exclusive currently only supported for random'

    return args

def process_method(args):
    method_split = re.split(r'[-+]', args.method)
    method_split.pop(0)

    for mode in method_split:
        if mode == 'x':
            args.input = mode
        elif mode == 'y':
            args.input = mode
        elif mode == 'psi':
            args.input = mode
        elif mode == 'binarize':
            args.binarize = True
        elif mode == 'normlize':
            args.normalize = False
        elif mode == 's':
            args.use_smatrix = True
        elif mode == 'e':
            args.use_ematrix = True
        elif mode == 'random':
            args.append_random = True

def get_random_seq(m, p_switch, k, seed, exclusive=False):
    rng = np.random.default_rng(seed)

    seq = np.zeros((m, k))
    if exclusive:
        transition_probs = [1 - p_switch] # keep label with p = 1-p_switch
        transition_probs.extend([p_switch/(k-1)]*(k-1)) # remaining transitions have sum to p_switch
        print(transition_probs)

        ind = np.empty(m)
        ind[0] = rng.choice(range(k), size = 1)
        for i in range(1, m):
            prev_label = ind[i-1]
            other_labels = list(range(k))
            other_labels.remove(prev_label)

            choices = [prev_label]
            choices.extend(other_labels)
            ind[i] = rng.choice(choices, p=transition_probs)
        print('ind', ind)
        for row, col in enumerate(ind):
            seq[row, int(col)] = 1
    else:
        seq[0, :] = rng.choice([1,0], size = k)
        for j in range(k):
            for i in range(1, m):
                if seq[i-1, j] == 1:
                    seq[i, j] = rng.choice([1,0], p=[1 - p_switch, p_switch])
                else:
                    seq[i, j] = rng.choice([1,0], p=[p_switch, 1 - p_switch])

    return seq

def get_PCA_split_seq(input, k):
    m, _ = input.shape
    pca = PCA()
    pca.fit(input)
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

def get_PCA_seq(input, k, normalize, use_kernel = False, kernel=None):
    '''
    Defines seq based on PCs of input.

    Inputs:
        input: matrix to perform PCA on
        k: numper of particle types / principal components to use
        normalize: True to normalize particle types / principal components to [-1, 1]
        use_kernel: True to use kernel PCA
        kernel: type of kernel to use

    Outputs:
        seq: array of particle types
    '''
    m, _ = input.shape
    if use_kernel:
        pca = KernelPCA(kernel = kernel)
        pca.fit(input)
        print(pca.eigenvalues_[:10])
    else:
        pca = PCA()
        pca.fit(input)
        print(pca.explained_variance_[:10])

    seq = np.zeros((m, k))
    for j in range(k):
        if use_kernel:
            pc = pca.eigenvectors_[:, j] # deprecated in 1.0
        else:
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

def get_k_means_seq(y, k, kr = True):
    m, _ = y.shape
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
    nmf = NMF(n_components = k, max_iter = 1000, init=None)
    nmf.fit(y)
    H = nmf.components_

    print(f"NMF reconstruction error: {nmf.reconstruction_err_}")

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
            print(f"WARNING: mark {mark} has insufficient coverage: {coverage}")
        seq_i = np.load(osp.join(data_folder, file))
        seq_i = seq_i[start:end+1, 1]
        seq[:, i] = seq_i
    i += 1
    if i < k:
        print(f"Warning: insufficient data - only {i} marks found")

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
        print(f"Mark {mark} has insufficient coverage: {coverage_arr[mark]}")
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
    Loads output from GNN model to use as ematrix or smatrix

    Inputs:
        model_path: path to model results
        sample: sample id (int)

    Outputs:
        s: np array of pairwise energies
    '''
    # determine model_type
    model_type = osp.split(osp.split(model_path)[0])[1]
    print(model_type)

    assert model_type == 'ContactGNNEnergy', f"Unrecognized model_type: {model_type}"
    energy_hat_path = osp.join(model_path, f"sample{sample}/energy_hat.txt")
    if osp.exists(energy_hat_path):
        energy = np.loadtxt(energy_hat_path)
    else:
        raise Exception(f's_path does not exist: {energy_hat_path}')

    return energy

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
        z_path = osp.join(model_path, f"sample{sample}/z.npy")
        if osp.exists(z_path):
            seq = np.load(z_path)
            assert seq.shape[1] == k
        else:
            raise Exception(f'z_path does not exist: {z_path}')
    elif model_type == 'ContactGNNEnergy':
        energy_hat_path = osp.join(model_path, "sample{sample}/energy_hat.txt")
        if osp.exists(energy_hat_path):
            energy_hat = np.loadtxt(energy_hat_path)
        else:
            raise Exception(f's_path does not exist: {energy_hat_path}')

        seq = get_PCA_seq(energy_hat, k, normalize)
    else:
        raise Exception(f"Unrecognized model_type: {model_type}")

    return seq

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
        lower_title = f'\nsilhouette score: {np.round(score, 3)}'
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
    print(args)

    if args.method.startswith('random'):
        seq = get_random_seq(args.m, args.p_switch, args.k, args.seed, args.exclusive)
    elif args.method == 'pca':
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        seq = get_PCA_seq(y_diag, args.k, args.normalize)
    elif args.method.startswith('pca_split'):
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        seq = get_PCA_split_seq(y_diag, args.k)
    elif args.method.startswith('kpca'):
        input_type = args.method.split('-')[1]
        if input_type.lower() == 'y':
            input = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        elif input_type.lower() == 'x':
            input = np.load(osp.join(args.sample_folder, 'x.npy'))[:args.m, :]
        elif input_type.lower() == 'psi':
            input = np.load(osp.join(args.sample_folder, 'psi.npy'))[:args.m, :]
        seq = get_PCA_seq(input, args.k, args.normalize, use_kernel = True, kernel = args.kernel)
    elif args.method.startswith('ground_truth'):
        x_file = osp.join(args.sample_folder, 'x.npy')
        psi_file = osp.join(args.sample_folder, 'psi.npy')
        if osp.exists(x_file):
            x = np.load(x_file)[:args.m, :]
            print(f'x loaded with shape {x.shape}')
        else:
            raise Exception(f'x not found for {args.sample_folder}')

        if osp.exists(psi_file):
            psi = np.load(psi_file)[:args.m, :]
            print(f'psi loaded with shape {psi.shape}')
        else:
            psi = x
            print(f'Warning: assuming x == psi for {args.sample_folder}')

        if args.input is None:
            assert args.use_ematrix or args.use_smatrix
        elif args.input == 'x':
            seq = x
            print(f'seq loaded with shape {seq.shape}')
        elif args.input == 'psi':
            seq = psi
            # this input will reproduce ground_truth-S barring random seed
            print(f'seq loaded with shape {seq.shape}')
        else:
            raise Exception(f'Unrecognized input mode {args.input} for method {args.method} for sample {args.sample_folder}')

        if args.append_random:
            assert not args.use_smatrix and not args.use_ematrix
            _, k = seq.shape
            assert args.k is not None
            assert args.k > k, f"{args.k} not > {k}"
            seq_random = get_random_seq(args.m, args.p_switch, args.k - k, args.seed)
            seq = np.concatenate((seq, seq_random), axis = 1)

        calc = False # TRUE if need to calculate e or s matrix
        if args.use_smatrix or args.use_ematrix:
            s_matrix_file = osp.join(args.sample_folder, 's_matrix.txt')
            if osp.exists(s_matrix_file):
                s = np.loadtxt(s_matrix_file)
            else:
                calc = True

            if args.use_ematrix:
                e_matrix_file = osp.join(args.sample_folder, 'e_matrix.txt')
                if osp.exists(e_matrix_file):
                    e = np.loadtxt(e_matrix_file)
                else:
                    calc = True

        if calc:
            chi = np.load(osp.join(args.sample_folder, 'chis.npy'))[:args.m, :]
            e, s = calculate_E_S(psi, chi)
    elif args.method.startswith('k_means'):
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        seq, args.labels = get_k_means_seq(y_diag, args.k)
        args.X = y_diag
    elif args.method.startswith('nmf'):
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]
        seq, args.labels = get_nmf_seq(args.m, y_diag, args.k, args.binarize)
        args.X = y_diag
    elif args.method.startswith('epigenetic'):
        seq, marks = get_epigenetic_seq(args.epigenetic_data_folder, args.k)
    elif args.method.startswith('chromhmm'):
        seq, labels = get_ChromHMM_seq(args.ChromHMM_data_file, args.k)
    elif args.method.startswith('gnn'):
        argparse_path = osp.join(args.model_path, 'argparse.txt')
        with open(argparse_path, 'r') as f:
            for line in f:
                if line == '--data_folder\n':
                    break
            data_folder = f.readline().strip()
            dataset = osp.split(data_folder)[1]
        assert dataset == args.dataset, f'Dataset mismatch: {dataset} vs { args.dataset}'

        if args.use_smatrix:
            s = get_energy_gnn(args.model_path, args.sample)
        elif args.use_ematrix:
            e = get_energy_gnn(args.model_path, args.sample)
        else:
            seq = get_seq_gnn(args.k, args.model_path, args.sample, args.normalize)
    else:
        raise Exception(f'Unkown method: {args.method}')

    if args.use_smatrix:
        m1, m2 = s.shape
        assert m1 == m2, f"shape mismatch, {m1} vs {m2}"
        assert m1 == args.m
        np.savetxt('s_matrix.txt', s, fmt = '%.3e')
        np.save('s.npy', s)
    elif args.use_ematrix:
        m1, m2 = e.shape
        assert m1 == m2, f"shape mismatch, {m1} vs {m2}"
        assert m1 == args.m
        np.savetxt('e_matrix.txt', e, fmt = '%.3e')
        np.save('e.npy', e)
        np.save('s.npy', s) # save s.npy either way
    else:
        m, k = seq.shape
        assert m == args.m, f"m mismatch: seq has {m} particles not {args.m}"
        if args.k is not None:
            assert k == args.k, f"k mismatch: seq has {k} particle types not {args.k} for method {args.method} for sample {args.sample_folder}"
        if args.save_npy:
            np.save('x.npy', seq)

    if args.plot:
        if args.use_smatrix:
            plotContactMap(s, 's_matrix.png', vmin = 'min', vmax = 'max', cmap = 'blue-red')
        elif args.use_ematrix:
            plotContactMap(e, 'e_matrix.png', vmin = 'min', vmax = 'max', cmap = 'blue-red')
        elif args.method in {'k_means', 'chromhmm'} or (args.method == 'nmf' and args.binarize):
            plot_seq_exclusive(seq, labels=args.labels, X=args.X)
        elif args.binarize:
            plot_seq_binary(seq)

### test functions ###
def test_nmf_k_means():
    args = getArgs()
    args.k = 4
    y_diag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
    args.X = y_diag

    seq, args.labels = get_nmf_seq(args.m, y_diag, args.k, binarize = True)
    plot_seq_exclusive(seq, labels=args.labels, X=args.X, show=True, save=False, title='nmf-binarize test')

    seq, args.labels = get_nmf_seq(args.m, y_diag, args.k, binarize = False)

    seq, args.labels = get_k_means_seq(y_diag, args.k, kr = True)
    plot_seq_exclusive(seq, labels=args.labels, X=args.X, show=True, save=False, title='k_means test')

def test_random():
    args = getArgs()

    args.k=4
    seq = get_random_seq(args.m, args.p_switch, args.k, args.seed, exclusive = True)

    plot_seq_exclusive(seq, show = True, save = False, title = 'test2')

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

    seq = get_seq_gnn(args.k, args.model_path, args.sample, args.normalize, args.use_smatrix)

def test_kPCA():
    args = getArgs()
    args.sample_folder = "/home/eric/sequences_to_contact_maps/dataset_11_14_21/samples/sample40"
    args.k = 4
    input = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]

    seq = get_PCA_seq(input, args.k, args.normalize, use_kernel = True, kernel = args.kernel)

    seq = get_PCA_seq(input, args.k, args.normalize)

def test_ground_truth_random():
    args = getArgs()
    args.k = 4
    args.m = 1000
    args.p_switch = 0.05

if __name__ ==  "__main__":
    main()
    # test_nmf_k_means()
    # test_random()
    # test_epi()
    # test_ChromHMM()
    # test_GNN()
    # test_kPCA()
