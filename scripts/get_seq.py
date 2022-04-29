import argparse
import os
import os.path as osp
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from knightRuiz import knightRuiz
from seq2contact import (LETTERS, DiagonalPreprocessing, R_pca,
                         clean_directories, crop, finalize_opt,
                         get_base_parser, get_dataset, load_E_S,
                         load_final_max_ent_S, load_saved_model, load_X_psi,
                         load_Y, plot_matrix, plot_seq_binary,
                         project_S_to_psi_basis, s_to_E, str2bool, str2int)
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA, KernelPCA
from sklearn.metrics import silhouette_score


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    seq_local = '..../sequences_to_contact_maps'
    chip_seq_data_local = osp.join(seq_local, 'chip_seq_data')
    # "./project2/depablo/erschultz/dataset_04_18_21"

    # input data args
    parser.add_argument('--data_folder', type=str, default=osp.join(seq_local,'dataset_01_15_22'),
                        help='location of input data')
    parser.add_argument('--sample', type=str, default='40',
                        help='sample id')
    parser.add_argument('--sample_folder', type=str,
                        help='location of input data')

    # standard args
    parser.add_argument('--method', type=str, default='k_means',
                        help='method for assigning particle types')
    parser.add_argument('--m', type=int, default=1024,
                        help='number of particles (will crop contact map) (-1 to infer)')
    parser.add_argument('--k', type=str2int, default=2,
                        help='sequences to generate')
    parser.add_argument('--scale_resolution', type=str2int, default=1,
                        help="generate seq at higher resolution, "
                            "find average frequency at lower resolution")
                        # TODO rename and document better

    # args for specific methods
    parser.add_argument('--seed', type=str2int,
                        help='random seed for numpy')
    parser.add_argument('--exclusive', type=str2bool, default=False,
                        help='True to use mutually exusive label (for random method)')
    parser.add_argument('--model_path', type=str,
                        help='path to GNN model')
    parser.add_argument('--epigenetic_data_folder', type=str,
                        default=osp.join(chip_seq_data_local, 'fold_change_control/processed'),
                        help='location of epigenetic data')
    parser.add_argument('--ChromHMM_data_file', type=str,
                        default=osp.join(chip_seq_data_local,
                                        'aligned_reads/ChromHMM_15/STATEBYLINE/'
                                        'HTC116_15_chr2_statebyline.txt'),
                        help='location of ChromHMM data')
    parser.add_argument('--p_switch', type=float, default=0.05,
                        help='probability to switch bead assignment (for method = random)')
    parser.add_argument('--kernel', type=str, default='poly',
                        help='kernel for kernel PCA')
    parser.add_argument('--local', type=str2bool, default=False,
                        help='True for local mode (relevant to method = GNN)')

    # post-processing args
    parser.add_argument('--save_npy', action='store_true',
                        help='true to save seq as .npy')
    parser.add_argument('--plot', action='store_true',
                        help='true to plot seq as .png')

    args = parser.parse_args()
    # below args are
    args.input = None # input in {y, x, psi}
    args.binarize = False # True to binarize labels (not implemented for all methods)') # TODO
    args.normalize = False # True to normalize labels to [0,1] (or [-1, 1] for some methods)
        # (not implemented for all methods)') # TODO
    args.use_ematrix = False
    args.use_smatrix = False
    args.append_random = False # True to append random seq
    args.load_chi = False # True to load e matrix learned from prior maxent and re-run
    args.project = False # True to project e into space of ground truth bead labels
        # (assumes load_chi is True)
    args.exp = False # (for RPCA) convert from log space back to original space
    args.diag = False # (for RPCA) apply diagonal processing
    args.rank = None # max rank for energy matrix
    args.method_copy = args.method
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
        elif args.method in {'nmf'} or args.method.startswith('block'):
            args.exclusive = True # will also be exclusive
        else:
            raise Exception(f'binarize not yet supported for {args.method}')
    if args.normalize:
        if args.method.startswith('pca') or args.method.startswith('rpca'):
            pass
        else:
            raise Exception(f'normalize not yet supported for {args.method}')
    if args.exclusive:
        if args.method in {'nmf'} or args.method.startswith('block'):
            args.binarize = True # will also be binary
        elif args.method in {'random'}:
            pass
        else:
            raise Exception(f'exclusive not yet supported for {args.method}')
    if args.scale_resolution != 1:
        assert args.method == 'random', f"{args.method} not supported yet"

    return args

def process_method(args):
    # for documentation, see comments in getArgs()
    method_split = re.split(r'[-+]', args.method)
    method_split.pop(0)
    modes = set(method_split)

    if 'x' in modes:
        args.input = 'x'
    if 'y' in modes:
        args.input = 'y'
    if 'psi' in modes:
        args.input = 'psi'
    if 'binarize' in modes:
        args.binarize = True
    if 'normalize' in modes:
        args.normalize = True
    if 's' in modes:
        args.use_smatrix = True
    if 'e' in modes:
        args.use_ematrix = True
        assert not args.use_smatrix
    if 'random' in modes:
        args.append_random = True
    if 'load_chi' in modes:
        args.load_chi = True
        assert args.use_ematrix or args.use_smatrix
    if 'project' in modes:
        assert args.use_ematrix or args.use_smatrix
        args.project = True
    if 'exp' in modes:
        args.exp = True
    if 'diag' in modes:
        args.diag = True

    for mode in modes:
        if mode.startswith('rank'):
            args.rank = int(mode[4])

### GetSeq class ###
class GetSeq():
    def __init__(self, m, k):
        self.m = m
        self.k = k

    def get_random_seq(self, p_switch, seed = None, exclusive=False, scale_resolution=1):
        rng = np.random.default_rng(seed)
        p_switch /= scale_resolution
        m = self.m * scale_resolution

        seq = np.zeros((m, self.k))
        if exclusive:
            transition_probs = [1 - p_switch] # keep label with p = 1-p_switch
            transition_probs.extend([p_switch/(self.k-1)]*(self.k-1))
            # remaining transitions have to sum to p_switch

            ind = np.empty(m)
            ind[0] = rng.choice(range(self.k), size = 1)
            for i in range(1, m):
                prev_label = ind[i-1]
                other_labels = list(range(self.k))
                other_labels.remove(prev_label)

                choices = [prev_label]
                choices.extend(other_labels)
                ind[i] = rng.choice(choices, p=transition_probs)
            for row, col in enumerate(ind):
                seq[row, int(col)] = 1
        else:
            seq[0, :] = rng.choice([1,0], size = self.k)
            for j in range(self.k):
                for i in range(1, m):
                    if seq[i-1, j] == 1:
                        seq[i, j] = rng.choice([1,0], p=[1 - p_switch, p_switch])
                    else:
                        seq[i, j] = rng.choice([1,0], p=[p_switch, 1 - p_switch])

        if scale_resolution != 1:
            seq_high_resolution = seq.copy()
            seq = np.zeros((self.m, self.k))
            for i in range(self.m):
                lower = i * scale_resolution
                upper = lower + scale_resolution
                slice = seq_high_resolution[lower:upper, :]
                sum = np.sum(slice, axis = 0)
                # sum /= scale_resolution
                sum /= np.sum(sum)
                seq[i, :] = np.nan_to_num(sum)

        return seq

    def get_block_seq(self, method):
        '''
        Method should be formatted likc "block-A100-B100"
        '''
        seq = np.zeros((self.m, self.k))
        method_split = re.split(r'[-+]', method)
        method_split.pop(0)
        lower_bead = 0
        letters = set()
        for s in method_split:
            letter = s[0].upper()
            letters.add(letter)
            label = LETTERS.find(letter)

            upper_bead = int(s[1:]) + lower_bead
            assert upper_bead <= self.m, f"too many beads: {upper_bead}"
            print(letter, lower_bead, upper_bead)

            seq[lower_bead:upper_bead, label] = 1
            lower_bead = upper_bead

        assert upper_bead == self.m, f"not enough beads: {upper_bead}"
        print(np.sum(seq, axis = 0))



        assert len(letters) == self.k, f"not enough letters ({letters}) for k = {self.k}"
        return seq

    def get_PCA_split_seq(self, input, normalize = False):
        input = crop(input, self.m)
        pca = PCA()
        pca.fit(input/np.std(input, axis = 0))
        seq = np.zeros((self.m, self.k))

        j = 0
        PC_count = self.k // 2 # 2 seqs per PC
        for pc_i in range(PC_count):
            pc = pca.components_[pc_i]

            pcpos = pc.copy()
            pcpos[pc < 0] = 0 # zero negative part
            if normalize:
                max = np.max(pcpos)
                # multiply by scale such that max x scale = 1
                scale = 1/max
                pcpos *= scale
            seq[:,j] = pcpos

            pcneg = pc.copy()
            pcneg[pc > 0] = 0 # zero positive part
            pcneg *= -1 # make positive
            if normalize:
                max = np.max(pcneg)
                # multiply by scale such that max x scale = 1
                scale = 1/max
                pcneg *= scale
            seq[:,j+1] = pcneg * -1


            j += 2
        return seq

    def get_PCA_seq(self, input, normalize = False, use_kernel = False, kernel = None):
        '''
        Defines seq based on PCs of input.

        Inputs:
            input: matrix to perform PCA on
            normalize: True to normalize particle types / principal components to [-1, 1]
            use_kernel: True to use kernel PCA
            kernel: type of kernel to use

        Outputs:
            seq: array of particle types
        '''
        input = crop(input, self.m)
        if use_kernel:
            assert kernel is not None
            pca = KernelPCA(kernel = kernel)
            pca.fit(input/np.std(input, axis = 0))
        else:
            pca = PCA()
            pca.fit(input/np.std(input, axis = 0))

        seq = np.zeros((self.m, self.k))
        for j in range(self.k):
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

    def get_RPCA_seq(self, input, normalize = False, exp = False, diag = False, max_it = 2000):
        '''
        Defines seq based on PCs of input.

        Inputs:
            input: matrix to perform PCA on
            normalize: True to normalize particle types / principal components to [-1, 1]

        Outputs:
            seq: array of particle types
        '''
        input = crop(input, self.m)
        input = input + 1e-8
        input = np.log(input)

        L, _ = R_pca(input).fit(max_iter=max_it)

        if exp:
            L = np.exp(L)
        if diag:
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(L)
            L = DiagonalPreprocessing.process(L, meanDist)

        return self.get_PCA_seq(L, normalize)

    def get_k_means_seq(self, y, kr = True):
        y = crop(y, self.m)

        if kr:
            yKR = np.log(knightRuiz(y))
        kmeans = KMeans(n_clusters = self.k)
        try:
            kmeans.fit(yKR)
        except ValueError as e:
            print(e)
            print('Not using KR')
            kmeans.fit(y)
        seq = np.zeros((self.m, self.k))
        seq[np.arange(self.m), kmeans.labels_] = 1
        return seq, kmeans.labels_

    def get_nmf_seq(self, input, binarize):
        '''
        Applies NMF to input.

        Inputs:
            input: input matrix
            binarize: True to binarize NMF result (is exclusive as well)

        Outputs:
            seq: bead types
            labels: categorical labels for bead types (None if binarize is True)
        '''
        input = crop(input, self.m)

        nmf = NMF(n_components = self.k, max_iter = 1000, init=None)
        nmf.fit(input)
        H = nmf.components_

        print(f"NMF reconstruction error: {nmf.reconstruction_err_}")

        if binarize:
            nmf.labels_ = np.argmax(H, axis = 0)
            seq = np.zeros((self.m, self.k))
            seq[np.arange(self.m), nmf.labels_] = 1
            return seq, nmf.labels_
        else:
            seq = H.T
            return seq, None

    def get_epigenetic_seq(self, data_folder, start=35000000, end=60575000,
                            res=25000, chr='2', min_coverage_prcnt=5):
        '''
        Loads experimental epigenetic data from data_folder to use as particle types.

        Inputs:
            data_folder: location of epigenetic data - file format: <chr>_*.npy
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
        assert m == self.m

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
        print(file_list[:self.k], )

        # choose k marks with most coverage
        seq = np.zeros((self.m, self.k))
        marks = []
        for i, (file, coverage) in enumerate(file_list[:self.k]):
            mark = file.split('_')[1]
            marks.append(mark)
            if coverage < min_coverage_prcnt / 100 * m:
                print(f"WARNING: mark {mark} has insufficient coverage: {coverage}")
            seq_i = np.load(osp.join(data_folder, file))
            seq_i = seq_i[start:end+1, 1]
            seq[:, i] = seq_i
        i += 1
        if i < self.k:
            print(f"Warning: insufficient data - only {i} marks found")

        return seq, marks

    def get_ChromHMM_seq(self, ifile, start=35000000, end=60575000, res=25000,
                        min_coverage_prcnt=5):
        start = int(start / res)
        end = int(end / res)
        m = end - start + 1 # number of particle in simulation
        assert m == self.m, f"m != m, {m} != {self.m}"

        with open(ifile, 'r') as f:
            f.readline()
            f.readline()
            states = np.array([int(state.strip()) - 1 for state in f.readlines()])
            # subtract 1 to convert to 0 based indexing
            states = states[start:end+1]


        seq = np.zeros((self.m, 15))
        seq[np.arange(self.m), states] = 1

        coverage_arr = np.sum(seq, axis = 0) # number of beads of each particle type

        # sort based on coverage
        insufficient_coverage = np.argwhere(coverage_arr < min_coverage_prcnt * m / 100).flatten()

        # exclude marks with no coverage
        for mark in insufficient_coverage:
            print(f"Mark {mark} has insufficient coverage: {coverage_arr[mark]}")
        seq = np.delete(seq, insufficient_coverage, 1)

        assert seq.shape[1] == self.k

        # get labels
        labels = np.where(seq == np.ones((m, 1)))[1]
        if len(labels) == m:
            # this is only true if all marks have sufficient coverage
            # i.e. none were deleted
            pass
        else:
            labels = None

        return seq, labels

    def get_seq_gnn(self, model_path, sample, normalize):
        # deprecated (seems to work but no promises)
        '''
        Loads output from GNN model to use as particle types, seq

        Inputs:
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
                assert seq.shape[1] == self.k
            else:
                raise Exception(f'z_path does not exist: {z_path}')
        elif model_type == 'ContactGNNEnergy':
            energy_hat_path = osp.join(model_path, f"sample{sample}/energy_hat.txt")
            if osp.exists(energy_hat_path):
                energy_hat = np.loadtxt(energy_hat_path)
            else:
                raise Exception(f's_path does not exist: {energy_hat_path}')

            seq = self.get_PCA_seq(energy_hat, normalize)
        else:
            raise Exception(f"Unrecognized model_type: {model_type}")

        return seq

# TODO
class GetChi():
    def __init__(self, k):
        self.k = k

def get_energy_gnn(model_path, sample_path, local, m):
        '''
        Loads output from GNN model to use as ematrix or smatrix

        Inputs:
            model_path: path to model results
            sample_path: path to sample

        Outputs:
            s: np array of pairwise energies
        '''
        print('\nget_energy_gnn')

        # extract sample info
        sample = osp.split(sample_path)[1]
        sample_id = int(sample[6:])
        sample_path_split = osp.normpath(sample_path).split(os.sep)
        sample_dataset = sample_path_split[-3]

        print(sample, sample_id, sample_dataset)

        # extract model info
        model_path_split = osp.normpath(model_path).split(os.sep)
        model_id = model_path_split[-1]
        model_type = model_path_split[-2]
        print(f'Model type: {model_type}')
        assert model_type == 'ContactGNNEnergy', f"Unrecognized model_type: {model_type}"

        argparse_path = osp.join(model_path, 'argparse.txt')
        with open(argparse_path, 'r') as f:
            for line in f:
                if line == '--data_folder\n':
                    break
            data_folder = f.readline().strip()
            gnn_dataset = osp.split(data_folder)[1]

        if gnn_dataset == sample_dataset:
            energy_hat_path = osp.join(model_path, f"{sample}/energy_hat.txt")
            if osp.exists(energy_hat_path):
                energy = np.loadtxt(energy_hat_path)
                return energy
        else:
            print(f'WARNING: dataset mismatch: {gnn_dataset} vs {sample_dataset}')

        # set up argparse options
        parser = get_base_parser()
        sys.argv = [sys.argv[0]] # delete args from get_seq, otherwise gnn opt will try and use them
        opt = parser.parse_args(['@{}'.format(argparse_path)])
        opt.id = int(model_id)
        opt.use_scratch = False # override use_scratch
        print(opt)
        opt = finalize_opt(opt, parser, local = local, debug = True)
        opt.m = m # override m
        opt.data_folder = osp.join('/',*sample_path_split[:-2]) # use sample_dataset not gnn_dataset
        opt.output_mode = None # don't need output, since only predicting
        opt.root_name = f'GNN{opt.id}-{sample}' # need this to be unique
        opt.log_file = sys.stdout # change
        print(opt)

        # get model
        model, _, _ = load_saved_model(opt, False)

        # get dataset
        dataset = get_dataset(opt, verbose = True, samples = [sample_id])

        # get prediction
        for i, data in enumerate(dataset):
            data = data.to(opt.device)
            yhat = model(data)
            yhat = yhat.cpu().detach().numpy()
            energy = yhat.reshape((opt.m,opt.m))

        # cleanup
        # opt.root is set in get_dataset
        clean_directories(GNN_path = opt.root)

        return energy

def plot_seq_continuous(seq, show = False, save = True, title = None):
    m, k = seq.shape
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = plt.cycler('color', cmap(ind))

    plt.figure(figsize=(6, 3))
    for i, c in enumerate(colors):
        plt.plot(np.arange(0, m), seq[:, i], label = i, color = c['color'])

    ax = plt.gca()
    if title is not None:
        plt.title(title, fontsize=16)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig('seq.png')
    plt.close()

def main():
    args = getArgs()
    print(args)
    if args.k == 0:
        return
    if args.m == -1:
        # infer m
        x, _ = load_X_psi(args.sample_folder, throw_exception = False)
        y, _ = load_Y(args.sample_folder, throw_exception = False)
        if x is not None:
            args.m, _ = x.shape
        elif y is not None:
            args.m, _ = y.shape

    getSeq = GetSeq(args.m, args.k)
    if args.load_chi:
        method_split = re.split(r'[-+]', args.method_copy)
        method_split.remove('load_chi')
        method_split.remove('E')
        if 'project' in method_split:
            method_split.remove('project')
        original_method = ''.join(method_split)
        replicate_path = osp.join(args.sample_folder, original_method, f'k{args.k}',
                                    'replicate1')
        assert osp.exists(replicate_path), f"path does not exist: {replicate_path}"
        s = load_final_max_ent_S(args.k, replicate_path, max_it_path = None)
        e = s_to_E(s)
    elif args.method.startswith('random'):
        seq = getSeq.get_random_seq(args.p_switch, args.seed, args.exclusive,
                                    args.scale_resolution)
    elif args.method.startswith('block'):
        seq = getSeq.get_block_seq(args.method)
    elif args.method.startswith('pca_split'):
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))
        seq = getSeq.get_PCA_split_seq(y_diag)
    elif args.method.startswith('pca'):
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))
        seq = getSeq.get_PCA_seq(y_diag, args.normalize)
    elif args.method.startswith('rpca'):
        L_file = osp.join(args.sample_folder, 'PCA_analysis', 'L_log.npy')
        if osp.exists(L_file):
            L = np.load(L_file)
            if args.exp:
                L = np.exp(L)
            if args.diag:
                meanDist = DiagonalPreprocessing.genomic_distance_statistics(L)
                L = DiagonalPreprocessing.process(L, meanDist)
            seq = getSeq.get_PCA_seq(L, args.normalize)
        else:
            y = np.load(osp.join(args.sample_folder, 'y.npy'))
            seq = getSeq.get_RPCA_seq(y, args.normalize, args.exp, args.diag)
    elif args.method.startswith('kpca'):
        if args.input == 'y':
            input = np.load(osp.join(args.sample_folder, 'y_diag.npy'))
        elif args.input == 'x':
            input = np.load(osp.join(args.sample_folder, 'x.npy'))
        elif args.input == 'psi':
            input = np.load(osp.join(args.sample_folder, 'psi.npy'))
        seq = getSeq.get_PCA_seq(input, args.normalize, use_kernel = True, kernel = args.kernel)
    elif args.method.startswith('ground_truth'):
        x, psi = load_X_psi(args.sample_folder, throw_exception = False)

        if args.input is None:
            assert args.use_ematrix or args.use_smatrix
        elif args.input == 'x':
            assert x is not None
            seq = x
            print(f'seq loaded with shape {seq.shape}')
        elif args.input == 'psi':
            assert psi is not None
            seq = psi
            # this input will reproduce ground_truth-S barring random seed
            print(f'seq loaded with shape {seq.shape}')
        else:
            raise Exception(f'Unrecognized input mode {args.input} for method {args.method} '
                            f'for sample {args.sample_folder}')

        if args.append_random:
            # TODO this may be broken
            assert not args.use_smatrix and not args.use_ematrix
            _, k = seq.shape
            assert args.k is not None
            assert args.k > k, f"{args.k} not > {k}"
            seq_random = GetSeq(args.m, args.k - k).get_random_seq(args.p_switch, args.seed)
            seq = np.concatenate((seq, seq_random), axis = 1)

        if args.use_smatrix or args.use_ematrix:
            e, s = load_E_S(args.sample_folder, psi)
    elif args.method.startswith('k_means') or args.method.startswith('k-means'):
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))
        seq, args.labels = getSeq.get_k_means_seq(y_diag)
        args.X = y_diag
    elif args.method.startswith('nmf'):
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))
        seq, args.labels = getSeq.get_nmf_seq(y_diag, args.binarize)
        args.X = y_diag
    elif args.method.startswith('epigenetic'):
        seq, marks = getSeq.get_epigenetic_seq(args.epigenetic_data_folder)
    elif args.method.startswith('chromhmm'):
        seq, labels = getSeq.get_ChromHMM_seq(args.ChromHMM_data_file)
    elif args.method.startswith('gnn'):
        if args.use_smatrix:
            s = get_energy_gnn(args.model_path, args.sample_folder, args.local, args.m)
        elif args.use_ematrix:
            s = get_energy_gnn(args.model_path, args.sample_folder, args.local, args.m)
            e = s_to_E(s)
        else:
            seq = getSeq.get_seq_gnn(args.model_path, args.sample, args.normalize)
    else:
        raise Exception(f'Unkown method: {args.method}')

    if args.use_smatrix or args.use_ematrix:
        if args.project:
            _, psi = load_X_psi(args.sample_folder)
            s, e = project_S_to_psi_basis(s, psi)
        s = crop(s, args.m)
        e = crop(e, args.m)
        if args.rank is not None:
            pca = PCA(n_components = args.rank)
            s_transform = pca.fit_transform((s+s.T)/2)
            print(s_transform.shape)
            print(f'Rank of S: {np.linalg.matrix_rank(s_transform)}')
            print(pca.components_.shape)
            s = pca.inverse_transform(s_transform)
            e = s_to_E(s)
        if args.use_smatrix:
            np.savetxt('s_matrix.txt', s, fmt = '%.3e')
            np.save('s.npy', s)
        elif args.use_ematrix:
            np.savetxt('e_matrix.txt', e, fmt = '%.3e')
            np.save('e.npy', e)
            np.save('s.npy', s)

        if args.m < 2000:
            print(f'Rank of S: {np.linalg.matrix_rank(s)}')
            print(f'Rank of E: {np.linalg.matrix_rank(e)}\n')
    else:
        m, k = seq.shape
        assert m == args.m, f"m mismatch: seq has {m} particles not {args.m}"
        if args.k is not None:
            assert k == args.k, f'''k mismatch: seq has {k} particle types not {args.k}
                                for method {args.method} for sample {args.sample_folder}'''
        if args.save_npy:
            np.save('x.npy', seq)

    if args.plot:
        if args.use_smatrix:
            plot_matrix(s, 's.png', vmin = 'min', vmax = 'max', cmap = 'blue-red', title = 'S')
        elif args.use_ematrix:
            plot_matrix(e, 'e.png', vmin = 'min', vmax = 'max', cmap = 'blue-red', title = 'E')
        elif args.method in {'k_means', 'chromhmm'} or (args.method == 'nmf' and args.binarize):
            plot_seq_exclusive(seq, labels=args.labels, X=args.X)
        elif args.binarize:
            plot_seq_binary(seq)

### Tester class ###
class Tester():
    def __init__(self):
        self.dataset = 'dataset_test'
        self.sample = 85
        self.sample_folder = osp.join('/home/eric', self.dataset, f'samples/sample{self.sample}')
        self.m = 300
        self.k = 3
        self.getSeq = GetSeq(self.m, self.k)

    def test_nmf_k_means(self):
        y_diag = np.load(osp.join(self.sample_folder, 'y_diag.npy'))

        seq, labels = self.getSeq.get_nmf_seq(y_diag, binarize = False)

        seq, labels = self.getSeq.get_nmf_seq(y_diag, binarize = True)
        plot_seq_exclusive(seq, labels = labels, X = y_diag, show = True,
                            save = False, title = 'nmf-binarize test')


        seq, labels = self.getSeq.get_k_means_seq(y_diag, kr = True)
        plot_seq_exclusive(seq, labels = labels, X = y_diag, show = True,
                            save = False, title = 'k_means test')

    def test_random(self):
        seq = self.getSeq.get_random_seq(p_switch=0.05, exclusive = True)
        # plot_seq_exclusive(seq, show = True, save = False, title = 'random-exclusive test')

        seq = self.getSeq.get_random_seq(p_switch=0.03, exclusive = False,
                                        scale_resolution = 25)
        plot_seq_continuous(seq, show = True, save = False,
                            title = 'random_scale_resolution test')

    def test_epi(self):
        args = getArgs()
        seq, marks = self.getSeq.get_epigenetic_seq(args.epigenetic_data_folder)
        print(marks)
        plot_seq_binary(seq, show = True, save = False, title = 'epi test',
                        labels = marks, x_axis = False)

    def test_ChromHMM(self):
        args = getArgs()
        k = 15
        y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))

        seq, labels = self.getSeq.get_ChromHMM_seq(args.ChromHMM_data_file, k,
                                                    min_coverage_prcnt = 0)
        plot_seq_exclusive(seq, labels=labels, X=y_diag, show=True, save=False,
                            title='ChromHMM test')

    def test_GNN(self):
        k = 2
        model_path = '/home/eric/sequences_to_contact_maps/results/ContactGNNEnergy/70'
        normalize = True

        seq = self.getSeq.get_seq_gnn(model_path, self.sample, normalize)

    def test_PCA(self):
        sample_folder = "/home/eric/dataset_test/samples/sample85"
        k = 4
        input = np.load(osp.join(sample_folder, 'y_diag.npy'))
        y = np.load(osp.join(sample_folder, 'y.npy'))

        seq = self.getSeq.get_PCA_seq(input, use_kernel = True, kernel = 'polynomial')
        # plot_seq_continuous(seq, show = True, save = False, title = 'kPCA test')

        seq = self.getSeq.get_PCA_seq(input, normalize = True)
        # plot_seq_continuous(seq, show = True, save = False, title = 'PCA-normalize test')

        seq = self.getSeq.get_RPCA_seq(y, normalize = True, max_it = 500)
        plot_seq_continuous(seq, show = True, save = False, title = 'RPCA test')


    def test_suite(self):
        # self.test_nmf_k_means()
        # self.test_random()
        # self.test_epi()
        # self.test_ChromHMM()
        # self.test_GNN()
        self.test_PCA()




if __name__ ==  "__main__":
    main()
    # Tester().test_suite()
