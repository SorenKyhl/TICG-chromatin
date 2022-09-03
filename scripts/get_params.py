import argparse
import json
import math
import os
import os.path as osp
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from knightRuiz import knightRuiz
from seq2contact import (LETTERS, ArgparserConverter, DiagonalPreprocessing,
                         InteractionConverter, R_pca, clean_directories, crop,
                         finalize_opt, get_base_parser, get_dataset, load_E_S,
                         load_final_max_ent_S, load_saved_model, load_X_psi,
                         load_Y, load_Y_diag, plot_matrix, plot_seq_binary,
                         plot_seq_exclusive, project_S_to_psi_basis, s_to_E)
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA, KernelPCA


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    # input data args
    parser.add_argument('--data_folder', type=str,
                        help='location of input data')
    parser.add_argument('--sample', type=str, default='40',
                        help='sample id')
    parser.add_argument('--sample_folder', type=str,
                        help='location of input data')

    # standard args

    parser.add_argument('--m', type=int, default=1024,
                        help='number of particles (will crop contact map) (-1 to infer)')
    parser.add_argument('--k', type=int, default=2,
                        help='sequences to generate')
    parser.add_argument('--plot', action='store_true',
                        help='true to plot seq as .png')

    # config args
    parser.add_argument('--config_ifile', type=str, default='default_config.json',
                        help='path to default config file')
    parser.add_argument('--config_ofile', type=str, default='config.json',
                            help='path to output config file')


    args, unknown = parser.parse_known_args()
    if args.sample_folder is None and args.data_folder is not None:
        args.sample_folder = osp.join(args.data_folder, 'samples', f'sample{args.sample}')

    with open(args.config_ifile, 'rb') as f:
        args.config = json.load(f)

    return args, unknown

class GetSeq():
    def __init__(self, args, unknown_args):
        self.m = args.m
        self.k = args.k
        self.sample_folder = args.sample_folder
        self.sample = args.sample
        self.plot = args.plot
        self._get_args(unknown_args)
        if self.args.method is None:
            return

        if self.args.use_ematrix or self.args.use_smatrix:
            # GetEnergy will handle this
            return

        self.set_up_seq()

    def _get_args(self, unknown_args):
        AC = ArgparserConverter()
        chip_seq_data_local = '../../sequences_to_contact_maps/chip_seq_data'
        parser = argparse.ArgumentParser(description='Seq parser')
        parser.add_argument('--method', type=AC.str2None,
                            help='method for assigning particle types')
        parser.add_argument('--seq_seed', type=AC.str2int,
                            help='random seed for numpy (None for random)')
        parser.add_argument('--exclusive', type=AC.str2bool, default=False,
                            help='True to use mutually exusive label (for random method)')
        parser.add_argument('--epigenetic_data_folder', type=str,
                            default=osp.join(chip_seq_data_local, 'fold_change_control/processed'),
                            help='location of epigenetic data')
        parser.add_argument('--ChromHMM_data_file', type=str,
                            default=osp.join(chip_seq_data_local,
                                            'aligned_reads/ChromHMM_15/STATEBYLINE/'
                                            'HTC116_15_chr2_statebyline.txt'),
                            help='location of ChromHMM data')
        parser.add_argument('--p_switch', type=AC.str2float, default=None,
                            help='probability to switch bead assignment (for method = random)')
        parser.add_argument('--lmbda', type=AC.str2float, default=0.8,
                            help='lambda for Markov matrix of method = random')
        parser.add_argument('--f', type=AC.str2float, default=0.5,
                            help='mark frequency for method = random')
        parser.add_argument('--kernel', type=str, default='poly',
                            help='kernel for kernel PCA')
        parser.add_argument('--scale_resolution', type=AC.str2int, default=1,
                            help="generate seq at higher resolution, "
                                "find average frequency at lower resolution")
                            # TODO rename and document better

        args, _ = parser.parse_known_args(unknown_args)
        print(args)
        self._process_method(args)

        args.labels = None
        args.X = None # X for silhouette_score

        self._check_args(args)
        self.args = args
        print("\nSeq args:")
        print(self.args)

    def _check_args(self, args):
        if args.method is None:
            return

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

    def _process_method(self, args):
        if args.method is None:
            return

        # default values
        args.input = None # input in {y, x, psi}
        args.binarize = False # True to binarize labels (not implemented for all methods)') # TODO
        args.normalize = False # True to normalize labels to [0,1] (or [-1, 1] for some methods)
            # (not implemented for all methods)') # TODO
        args.scale = False # True to scale variance for PCA
        args.use_ematrix = False
        args.use_smatrix = False
        args.append_random = False # True to append random seq
        args.exp = False # (for RPCA) convert from log space back to original space
        args.diag = False # (for RPCA) apply diagonal processing
        args.add_constant = False # add constant seq (all ones)


        args.method = args.method.lower()
        args.method_copy = args.method
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
        if 'scale' in modes:
            args.scale = True
        if 's' in modes:
            args.use_smatrix = True
        if 'e' in modes:
            args.use_ematrix = True
            assert not args.use_smatrix
        if 'random' in modes:
            args.append_random = True
        if 'exp' in modes:
            args.exp = True
        if 'diag' in modes:
            args.diag = True
        if 'constant' in modes:
            args.add_constant = True
            self.k -= 1

    def get_random_seq(self, lmbda = None, f = None, p_switch = None,
                        seed = None, exclusive = False, scale_resolution = 1):
        rng = np.random.default_rng(seed)
        assert (p_switch is not None) ^ (lmbda is not None)

        m = self.m * scale_resolution
        seq = np.zeros((m, self.k))

        if p_switch is not None:
            p_switch /= scale_resolution

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
        else:
            assert scale_resolution == 1, 'not supported yet'
            # lmda is not None
            if f is None:
                f = 0.5
                print('Using f = 0.5')

            seq[0, :] = rng.choice([1,0], size = self.k)
            for j in range(self.k):
                for i in range(1, m):
                    if seq[i-1, j] == 1:
                        p11 = f*(1-lmbda)+lmbda
                        seq[i, j] = rng.choice([1,0], p = [p11, 1 - p11])
                    else:
                        # equals 0
                        p00 = f*(lmbda-1) + 1
                        seq[i, j] = rng.choice([1,0], p = [1 - p00, p00])

        return seq

    def get_block_seq(self, method):
        '''
        Method should be formatted like "block-A100-B100"
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
        pca.fit(input)
        # /np.std(input, axis = 0)
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

    def get_PCA_seq(self, input, normalize = False, scale = False,
                    use_kernel = False, kernel = None):
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
        else:
            pca = PCA()

        if scale:
            pca.fit(input/np.std(input, axis = 0))
        else:
            pca.fit(input)

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

    def get_RPCA_seq(self, input, normalize = False, scale = False, exp = False,
                        diag = False, max_it = 2000):
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

        return self.get_PCA_seq(L, normalize, scale)

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

    def get_seq_gnn(self, model_path, sample, normalize, scale):
        # deprecated
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

            seq = self.get_PCA_seq(energy_hat, normalize, scale)
        else:
            raise Exception(f"Unrecognized model_type: {model_type}")

        return seq

    def set_up_seq(self):
        args = self.args
        if self.k == 0:
            return
        if self.m == -1:
            # infer m
            x, _ = load_X_psi(self.sample_folder, throw_exception = False)
            y, _ = load_Y(self.sample_folder, throw_exception = False)
            if x is not None:
                self.m, _ = x.shape
            elif y is not None:
                self.m, _ = y.shape

        elif args.method is None:
            return
        elif args.method == 'pca-soren':
            files = [osp.join(self.sample_folder, f'pcf{i}.txt') for i in range(1,6)]
            seq = np.zeros((self.m, self.k))
            for i, f in enumerate(files):
                seq[:, i] = np.loadtxt(f)
        elif args.method.startswith('random'):
            seq = self.get_random_seq(args.lmbda, args.f, args.p_switch, args.seq_seed, args.exclusive,
                                        args.scale_resolution)
        elif args.method.startswith('block'):
            seq = self.get_block_seq(args.method)
        elif args.method.startswith('pca_split'):
            y_diag = load_Y_diag(self.sample_folder)
            seq = self.get_PCA_split_seq(y_diag)
        elif args.method.startswith('pca'):
            y_diag = load_Y_diag(self.sample_folder)
            seq = self.get_PCA_seq(y_diag, args.normalize, args.scale)
        elif args.method.startswith('rpca'):
            L_file = osp.join(self.sample_folder, 'PCA_analysis', 'L_log.npy')
            if osp.exists(L_file):
                L = np.load(L_file)
                if args.exp:
                    L = np.exp(L)
                if args.diag:
                    meanDist = DiagonalPreprocessing.genomic_distance_statistics(L)
                    L = DiagonalPreprocessing.process(L, meanDist)
                seq = self.get_PCA_seq(L, args.normalize, args.scale)
            else:
                y = np.load(osp.join(self.sample_folder, 'y.npy'))
                seq = self.get_RPCA_seq(y, args.normalize, args.exp, args.diag)
        elif args.method.startswith('kpca'):
            if args.input == 'y':
                input = load_Y_diag(self.sample_folder)
            elif args.input == 'x':
                input = np.load(osp.join(self.sample_folder, 'x.npy'))
            elif args.input == 'psi':
                input = np.load(osp.join(self.sample_folder, 'psi.npy'))
            seq = self.get_PCA_seq(input, args.normalize, args.scale, use_kernel = True, kernel = args.kernel)
        elif args.method.startswith('ground_truth'):
            x, psi = load_X_psi(self.sample_folder, throw_exception = False)

            if args.input is None:
                assert args.use_ematrix or args.use_smatrix, 'missing input'
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
                                f'for sample {self.sample_folder}')

            if args.append_random:
                # TODO this is broken
                assert not args.use_smatrix and not args.use_ematrix
                _, k = seq.shape
                assert args.k > k, f"{args.k} not > {k}"
                seq_random = GetSeq(args.m, args.k - k).get_random_seq(args.lmbda, args.f, args.p_switch, args.seq_seed)
                seq = np.concatenate((seq, seq_random), axis = 1)

            if args.use_smatrix or args.use_ematrix:
                e, s = load_E_S(self.sample_folder, psi)
        elif args.method.startswith('k_means') or args.method.startswith('k-means'):
            y_diag = np.load(osp.join(self.sample_folder, 'y_diag.npy'))
            seq, args.labels = self.get_k_means_seq(y_diag)
            args.X = y_diag
        elif args.method.startswith('nmf'):
            y_diag = np.load(osp.join(self.sample_folder, 'y_diag.npy'))
            seq, args.labels = self.get_nmf_seq(y_diag, args.binarize)
            args.X = y_diag
        elif args.method.startswith('epigenetic'):
            seq, marks = self.get_epigenetic_seq(args.epigenetic_data_folder)
        elif args.method.startswith('chromhmm'):
            seq, labels = self.get_ChromHMM_seq(args.ChromHMM_data_file)
        else:
            raise Exception(f'Unkown method: {args.method}')

        m, k = seq.shape
        assert m == self.m, f"m mismatch: seq has {m} particles not {self.m}"
        if self.k > 0:
            assert k == self.k, f'''k mismatch: seq has {k} particle types not {self.k}
                                for method {args.method} for sample {self.sample_folder}'''

        if args.add_constant:
            seq = np.concatenate((seq, np.ones((m, 1))), axis = 1)

        np.save('x.npy', seq)

        if self.plot:
            if args.method in {'k_means', 'chromhmm'} or (args.method == 'nmf' and args.binarize):
                plot_seq_exclusive(seq, labels=args.labels, X=args.X)
            elif args.binarize:
                plot_seq_binary(seq)

class GetPlaidChi():
    def __init__(self, args, unknown_args):
        self.k = args.k
        self.m = args.m
        self._get_args(unknown_args)
        self.set_up_plaid_chi()

    def _get_args(self, unknown_args):
        AC = ArgparserConverter()
        parser = argparse.ArgumentParser(description='Plaid chi parser')
        parser.add_argument('--chi', type=AC.str2list2D,
                            help='chi matrix using latex separator style'
                                '(None to generate chi with chi_method)')
        parser.add_argument('--chi_method', type=AC.str2None, default='random',
                            help='method for generating chi if not given')
        parser.add_argument('--min_chi', type=float, default=-1.,
                            help='minimum chi value for random generation')
        parser.add_argument('--max_chi', type=float, default=1.,
                            help='maximum chi value for random generation')
        parser.add_argument('--fill_diag', type=AC.str2float,
                            help='fill diag of chi with given value (None to skip)')
        parser.add_argument('--fill_offdiag', type=AC.str2float,
                            help='fill off diag of chi with given value (None to skip)')
        parser.add_argument('--ensure_distinguishable', action='store_true',
                            help='true to ensure that corresponding psi is distinguishable')
        parser.add_argument('--chi_seed', type=AC.str2int,
                            help='seed for generating chi (None for random)')
        parser.add_argument('--chi_constant', type=AC.str2float, default=0,
                            help='constant to add to chi')
        parser.add_argument('--chi_multiplier', type=AC.str2float, default=1,
                            help='multiplier to multiply by chi')

        self.args, _ = parser.parse_known_args(unknown_args)
        print('\nChi args')
        print(self.args)

    def set_up_plaid_chi(self):
        args = self.args
        if self.k == 0:
            print('k is 0, returning')
            return
        elif self.k == -1:
            x = np.load('x.npy')
            _, self.k = x.shape
            print(f'k is -1, inferred {self.k}')

        if args.chi is not None:
            chi = np.triu(args.chi) # zero lower triangle
            rows, cols = chi.shape
            assert self.k == rows, f'number of particle types {self.k} does not match shape of chi {rows}'
            assert rows == cols, f"chi not square: {chi}"
            conv = InteractionConverter(self.k, chi)
            if not GetPlaidChi.unique_rows(conv.getS()):
                print('Warning: particles are not distinguishable')
        elif args.chi_method is None:
            chi = None
        elif args.chi_method == 'random':
            chi = self.random_chi()
        elif args.chi_method.startswith('polynomial'):
            x = np.load('x.npy') # original particle types that interact nonlinearly
            ind = np.triu_indices(self.k)
            self.k = int(self.k*(self.k+1)/2)
            psi = np.zeros((self.m, self.k))
            for i in range(self.m):
                psi[i] = np.outer(x[i], x[i])[ind]

            np.save('psi.npy', psi)
            chi = self.random_chi()
        else:
            raise Exception(f"Unrecognized chi_method: {args.chi_method}")


        # save chi
        if chi is not None:
            chi += args.chi_constant
            chi *= args.chi_multiplier
            print(f'Rank of chi: {np.linalg.matrix_rank(chi)}')
            np.save('chis.npy', chi)

    def _generate_random_chi(self, rng = np.random.default_rng(), decimals = 1):
        '''Initializes random chi array.'''
        args = self.args

        # create array with random values in [minval, maxVal]
        rands = rng.random(size=(self.k, self.k)) * (args.max_chi - args.min_chi) + args.min_chi

        # zero lower triangle
        chi = np.triu(rands)

        if args.fill_offdiag is not None:
            # fills off diag chis with value of fill_offdiag
            diag_chi = np.diagonal(chi)
            chi = np.ones((args.k, args.k)) * args.fill_offdiag
            di = np.diag_indices(args.k)
            chi[di] = diag_chi
        if args.fill_diag is not None:
            # fills diag chis with value of fill_diag
            di = np.diag_indices(args.k)
            chi[di] = args.fill_diag

        return np.round(chi, decimals = decimals)

    def random_chi(self):
        args = self.args
        rng = np.random.default_rng(args.chi_seed)
        chi = self._generate_random_chi(rng)
        if args.ensure_distinguishable and self.k < 10: # if k is too large this is too RAM intensive
            conv = InteractionConverter(self.k, chi)
            max_it = 10
            it = 0
            while not GetPlaidChi.unique_rows(conv.getS()) and it < max_it: # defaults to False
                # generate random chi
                conv.chi = _generate_random_chi(rng)
                it += 1
            if it == max_it:
                print('Warning: maximum iteration reached')
                print('Warning: particles are not distinguishable')
            chi = conv.chi

        return chi

    @staticmethod
    def unique_rows(array):
        if array is None:
            return False

        if len(np.unique(array, axis=0)) / len(array) == 1.:
            return True
        else:
            return False

class GetDiagChi():
    def __init__(self, args, unknown_args):
        self.sample_folder = args.sample_folder
        self.m = args.m
        self.config = args.config
        self._get_args(unknown_args)
        self.set_up_diag_chi()

        self.config['diag_cutoff'] = self.args.diag_cutoff
        self.config['diag_start'] = self.args.diag_start

    def _get_args(self, unknown_args):
        AC = ArgparserConverter()
        parser = argparse.ArgumentParser(description='Diag chi parser')
        parser.add_argument('--diag_chi', type=AC.str2list,
                            help='diag chi (None to generate via diag_chi_method)')
        parser.add_argument('--diag_bins', type=AC.str2int, default=20,
                            help='number of diagonal bins for diag_chi_method')
        parser.add_argument('--diag_chi_method', type=AC.str2None, default='linear',
                            help='method for generating diag_chi if not given'
                                '(None for no diag_chi)')
        parser.add_argument('--diag_chi_slope', type=float, default=1.,
                            help='slope (in thousandths) for diag_chi_method = log')
        parser.add_argument('--dense_diagonal_on', type=AC.str2bool, default=False,
                            help='True to place 1/2 of beads left of cutoff')
        parser.add_argument('--dense_diagonal_cutoff', type=AC.str2float, default=1/16,
                            help='cutoff = nbeads * dense_diag_cutoff')
        parser.add_argument('--dense_diagonal_loading', type=AC.str2float, default=0.5,
                            help='proportion of beads to place left of cutoff')
        parser.add_argument('--small_binsize', type=int, default=0,
                            help='specify small_binsize instead of using dense_diagonal_cutoff')
        parser.add_argument('--big_binsize', type=int, default=-1,
                            help='specify big_binsize instead of using dense_diagonal_cutoff')
        parser.add_argument('--n_small_bins', type=int, default=0,
                            help='specify n_small_bins instead of using dense_diagonal_loading')
        parser.add_argument('--n_big_bins', type=int, default=-1,
                            help='specify n_big_bins instead of using dense_diagonal_loading')
        parser.add_argument('--max_diag_chi', type=float, default=0,
                            help='maximum diag chi value for diag_chi_method')
        parser.add_argument('--diag_chi_constant', type=AC.str2float, default=0,
                            help='constant to add to chi diag')
        parser.add_argument('--mlp_model_path', type=str,
                            help='path to MLP model')
        parser.add_argument('--m_continuous', type=AC.str2int,
                            help='Use m larger than self.m to define diag chis then crop')
        parser.add_argument('--diag_start', type=int, default=0,
                            help='minimum d to use diag chi')
        parser.add_argument('--diag_cutoff', type=AC.str2int,
                            help='maximum d to use diag chi (None to ignore)')

        self.args, _ = parser.parse_known_args(unknown_args)
        if self.args.m_continuous is None:
            self.args.m_continuous = self.m
        if self.args.diag_cutoff is None:
            self.args.diag_cutoff = self.m
        print('\nDiag chi args:')
        print(self.args)

    def set_up_diag_chi(self):
        args = self.args
        if args.diag_chi_method is not None:
            args.diag_chi_method = args.diag_chi_method.lower()
            d_arr = np.arange(args.m_continuous)
            args.diag_chi_slope /= 1000
            if args.diag_chi_method == 'linear':
                diag_chis_continuous = np.linspace(0, args.max_diag_chi, args.m_continuous)
            elif args.diag_chi_method == 'log':
                scale = args.max_diag_chi / np.log(args.diag_chi_slope * (args.m_continuous - 1) + 1)
                diag_chis_continuous = scale * np.log(args.diag_chi_slope * d_arr + 1)
            elif args.diag_chi_method == 'logistic':
                diag_chis_continuous = 2 * args.max_diag_chi / (1 + np.exp(-args.diag_chi_slope * d_arr)) - args.max_diag_chi
            elif args.diag_chi_method == 'exp':
                diag_chis_continuous = args.max_diag_chi - 1.889 * np.exp(-args.diag_chi_slope * d_arr)
            elif args.diag_chi_method == 'mlp':
                diag_chis = self.get_diag_chi_mlp(args.mlp_model_path, self.sample_folder)
                assert len(diag_chis) == args.diag_bins, f"Shape mismatch: {len(diag_chis)} vs {args.diag_bins}"
            else:
                raise Exception(f'Unrecognized chi diag method {args.diag_chi_method}')

            if diag_chis_continuous is not None:
                diag_chis = self.coarse_grain_diag_chi(diag_chis_continuous)
                np.save('diag_chis_continuous', diag_chis_continuous)

        elif args.diag_chi is not None:
            diag_chis_continuous = None
            diag_chis = np.array(args.diag_chi)
        else:
            return

        diag_chis += args.diag_chi_constant
        print('diag_chis: ', diag_chis, diag_chis.shape)
        np.save('diag_chis', diag_chis)
        self.config["diag_chis"] = list(diag_chis) # ndarray not json serializable

    def coarse_grain_diag_chi(self, diag_chis_continuous):
        args = self.args
        m_eff = args.diag_cutoff - args.diag_start # number of beads with nonzero interaction

        # get bin sizes
        if args.dense_diagonal_on:
            if args.dense_diagonal_loading is not None:
                n_small_bins = int(args.dense_diagonal_loading * args.diag_bins)
                assert args.diag_bins > n_small_bins, f"{args.diag_bins} < {n_small_bins}"
                n_big_bins = args.diag_bins - n_small_bins
            else:
                n_small_bins = args.n_small_bins
                n_big_bins = args.n_big_bins

            if args.dense_diagonal_cutoff is not None:
                dividing_line = m_eff * args.dense_diagonal_cutoff
                small_binsize = int(dividing_line / (n_small_bins))
                big_binsize = int((m_eff- dividing_line) / n_big_bins)
            else:
                small_binsize = args.small_binsize
                dividing_line = small_binsize * n_small_bins
                big_binsize = args.big_binsize

            if n_big_bins == -1:
                remainder = m_eff - dividing_line
                n_big_bins = math.floor(args.diag_bins - n_small_bins)
                while remainder % n_big_bins != 0 and n_big_bins < remainder:
                    n_big_bins += 1

                big_binsize = remainder // n_big_bins
                if n_small_bins + n_big_bins != args.diag_bins:
                    print(f'args.diag_bins changed from {args.diag_bins} to {n_small_bins + n_big_bins}')
                    args.diag_bins = n_small_bins + n_big_bins


            assert n_small_bins * small_binsize + n_big_bins * big_binsize == m_eff, f'{n_small_bins}x{small_binsize} + {n_big_bins}x{big_binsize} != {m_eff}'
            print(f'{n_small_bins}x{small_binsize} + {n_big_bins}x{big_binsize} = {m_eff}')

            self.config['n_small_bins'] = n_small_bins
            self.config['n_big_bins'] = n_big_bins
            self.config['small_binsize'] = small_binsize
            self.config['big_binsize'] = big_binsize
        else:
            binsize = self.m / args.diag_bins
            self.config['n_small_bins'] = 0
            self.config['n_big_bins'] = args.diag_bins
            self.config['small_binsize'] = 0
            self.config['big_binsize'] = binsize

        # get diag chis
        i = 0
        diag_chis = np.zeros(args.diag_bins)
        curr_bin_vals = []
        prev_bin = 0
        for d, val in enumerate(diag_chis_continuous[args.diag_start:args.diag_cutoff]):
            if args.dense_diagonal_on:
                if d > dividing_line:
                    bin = n_small_bins + math.floor((d - dividing_line) / big_binsize)
                else:
                    bin =  math.floor(d / small_binsize)
            else:
                bin = int(d / binsize)

            curr_bin = bin
            if curr_bin != prev_bin:
                prev_bin = curr_bin
                diag_chis[i] = np.mean(curr_bin_vals)
                curr_bin_vals = []
                i += 1
                if i >= len(diag_chis):
                    return diag_chis
            curr_bin_vals.append(val)
        diag_chis[i] = np.mean(curr_bin_vals)

        return diag_chis


    def get_diag_chi_mlp(self, model_path, sample_path):
        print('\nget_diag_chi_mlp')

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
        assert model_type == 'MLP', f"Unrecognized model_type: {model_type}"

        argparse_path = osp.join(model_path, 'argparse.txt')
        with open(argparse_path, 'r') as f:
            for line in f:
                if line == '--data_folder\n':
                    break
            data_folder = f.readline().strip()
            mlp_dataset = osp.split(data_folder)[1]

        if mlp_dataset == sample_dataset:
            hat_path = osp.join(model_path, f"{sample}/diag_chi_hat.txt")
            if osp.exists(hat_path):
                diag_chi = np.loadtxt(hat_path)
                return diag_chi
        else:
            print(f'WARNING: dataset mismatch: {mlp_dataset} vs {sample_dataset}')

        # set up argparse options
        parser = get_base_parser()
        sys.argv = [sys.argv[0]] # delete args from get_params, otherwise gnn opt will try and use them
        opt = parser.parse_args(['@{}'.format(argparse_path)])
        opt.id = int(model_id)
        print(opt)
        opt = finalize_opt(opt, parser, local = True, debug = True)
        opt.data_folder = osp.join('/',*sample_path_split[:-2]) # use sample_dataset not mlp_dataset
        opt.log_file = sys.stdout # change
        print(opt)

        # get model
        model, _, _ = load_saved_model(opt, False)

        # get dataset
        dataset = get_dataset(opt, verbose = True, samples = [sample_id])
        print(dataset)

        # get prediction
        for i, (x, y) in enumerate(dataset):
            x = x.to(opt.device)
            print('x', x, x.shape)
            yhat = model(x)
            yhat = yhat.cpu().detach().numpy()
            diag_chi = yhat.reshape((-1)).astype(np.float64)

        return diag_chi

class GetEnergy():
    def __init__(self, args, unknown_args):
        self.m = args.m
        self.sample_folder = args.sample_folder
        self.plot = args.plot
        self._get_args(unknown_args)

        if self.args.method is None:
            return

        if not self.args.use_ematrix and not self.args.use_smatrix:
            # should have been handled by GetSeq already
            return

        self.set_up_energy()

    def _get_args(self, unknown_args):
        AC = ArgparserConverter()
        parser = argparse.ArgumentParser(description='Energy parser')
        parser.add_argument('--method', type=AC.str2None,
                            help='method for assigning energy')
        parser.add_argument('--gnn_model_path', type=str,
                            help='path to GNN model')


        self.args, _ = parser.parse_known_args(unknown_args)
        self._process_method(self.args)
        print('\nEnergy args')
        print(self.args)

    def _process_method(self, args):
        if args.method is None:
            return

        # default values
        args.use_ematrix = False
        args.use_smatrix = False
        args.load_maxent = False # True to load e matrix learned from prior maxent and re-run
        args.project = False # True to project e into space of ground truth bead labels
            # (assumes load_maxent is True)
        args.rank = None # max rank for energy matrix

        args.method = args.method.lower()
        args.method_copy = args.method
        method_split = re.split(r'[-+]', args.method)
        method_split.pop(0)
        modes = set(method_split)

        if 's' in modes:
            args.use_smatrix = True
        if 'e' in modes:
            args.use_ematrix = True
            assert not args.use_smatrix
        if 'load_maxent' in modes:
            args.load_maxent = True
            assert args.use_ematrix or args.use_smatrix
        if 'project' in modes:
            assert args.use_ematrix or args.use_smatrix
            args.project = True

        for mode in modes:
            if mode.startswith('rank'):
                args.rank = int(mode[4])

    def set_up_energy(self):
        args = self.args
        if args.load_maxent:
            method_split = re.split(r'[-+]', args.method_copy)
            method_split.remove('load_maxent')
            method_split.remove('e')
            if 'project' in method_split:
                method_split.remove('project')
            original_method = ''.join(method_split)
            replicate_path = osp.join(self.sample_folder, original_method, f'k{args.k}',
                                        'replicate1')
            assert osp.exists(replicate_path), f"path does not exist: {replicate_path}"
            s = load_final_max_ent_S(args.k, replicate_path, max_it_path = None)
            e = s_to_E(s)
        elif args.method.startswith('ground_truth'):
            e, s = load_E_S(self.sample_folder)
        elif args.method.startswith('gnn'):
            if args.use_smatrix:
                s = self.get_energy_gnn(args.gnn_model_path, self.sample_folder)
            if args.use_ematrix:
                s = self.get_energy_gnn(args.gnn_model_path, self.sample_folder)
                e = s_to_E(s)
        else:
            raise Exception(f'Unkown method: {args.method}')

        if args.project:
            _, psi = load_X_psi(self.sample_folder)
            s, e = project_S_to_psi_basis(s, psi)
        s = crop(s, self.m)
        e = crop(e, self.m)
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

        if self.m < 2000:
            print(f'Rank of S: {np.linalg.matrix_rank(s)}')
            print(f'Rank of E: {np.linalg.matrix_rank(e)}\n')


        if self.plot:
            if args.use_smatrix:
                plot_matrix(s, 's.png', vmin = 'min', vmax = 'max', cmap = 'blue-red', title = 'S')
            elif args.use_ematrix:
                plot_matrix(e, 'e.png', vmin = 'min', vmax = 'max', cmap = 'blue-red', title = 'E')

    def get_energy_gnn(self, model_path, sample_path):
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
        sys.argv = [sys.argv[0]] # delete args from get_params, otherwise gnn opt will try and use them
        opt = parser.parse_args(['@{}'.format(argparse_path)])
        opt.id = int(model_id)
        print(opt)
        opt = finalize_opt(opt, parser, local = True, debug = True)
        if self.m > 0:
            opt.m = self.m # override m
        opt.data_folder = osp.join('/',*sample_path_split[:-2]) # use sample_dataset not gnn_dataset
        opt.output_mode = None # don't need output, since only predicting
        opt.root_name = f'GNN{opt.id}-{sample}' # need this to be unique
        opt.log_file = sys.stdout # change
        opt.cuda = False # force to use cpu
        opt.device = torch.device('cpu')
        print(opt)

        # get model
        model, _, _ = load_saved_model(opt, False)

        # get dataset
        dataset = get_dataset(opt, verbose = True, samples = [sample_id])
        print(dataset)

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
    args, unknown = getArgs()
    print(args)
    GetSeq(args, unknown)
    GetPlaidChi(args, unknown)
    GetDiagChi(args, unknown)
    GetEnergy(args, unknown)

    with open(args.config_ofile, 'w') as f:
        json.dump(args.config, f, indent = 2)

### Tester class ###
class Tester():
    def __init__(self):
        self.dataset = 'dataset_test'
        self.sample = 1
        self.sample_folder = osp.join('/home/erschultz', self.dataset, f'samples/sample{self.sample}')
        self.m = 1024
        self.k = 2
        self.self = self(self, None)

    def test_nmf_k_means(self):
        y_diag = np.load(osp.join(self.sample_folder, 'y_diag.npy'))

        # seq, labels = self.self.get_nmf_seq(y_diag, binarize = False)
        #
        # seq, labels = self.self.get_nmf_seq(y_diag, binarize = True)
        # plot_seq_exclusive(seq, labels = labels, X = y_diag, show = True,
        #                     save = False, title = 'nmf-binarize test')


        seq, labels = self.self.get_k_means_seq(y_diag)
        plot_seq_exclusive(seq, labels = labels, X = y_diag, show = True,
                            save = False, title = 'k_means test')

    def test_random(self):
        seq = self.self.get_random_seq(lmbda=0.8, exclusive = False)
        plot_seq_continuous(seq, show = True, save = False,
                            title = 'random_lmbda_resolution test')

        seq = self.self.get_random_seq(p_switch=0.05, exclusive = True)
        plot_seq_exclusive(seq, show = True, save = False, title = 'random-exclusive test')

        seq = self.self.get_random_seq(p_switch=0.03, exclusive = False,
                                        scale_resolution = 25)
        plot_seq_continuous(seq, show = True, save = False,
                            title = 'random_scale_resolution test')

    def test_epi(self):
        args = getArgs()
        seq, marks = self.self.get_epigenetic_seq(args.epigenetic_data_folder)
        print(marks)
        plot_seq_binary(seq, show = True, save = False, title = 'epi test',
                        labels = marks, x_axis = False)

    def test_ChromHMM(self):
        args = getArgs()
        k = 15
        y_diag = np.load(osp.join(self.sample_folder, 'y_diag.npy'))

        seq, labels = self.self.get_ChromHMM_seq(args.ChromHMM_data_file, k,
                                                    min_coverage_prcnt = 0)
        plot_seq_exclusive(seq, labels=labels, X=y_diag, show=True, save=False,
                            title='ChromHMM test')

    def test_GNN(self):
        k = 2
        model_path = '/home/eric/sequences_to_contact_maps/results/ContactGNNEnergy/70'
        normalize = True

        seq = self.self.get_seq_gnn(model_path, self.sample, normalize)

    def test_PCA(self):
        sample_folder = "/home/eric/dataset_test/samples/sample85"
        k = 4
        input = np.load(osp.join(sample_folder, 'y_diag.npy'))
        y = np.load(osp.join(sample_folder, 'y.npy'))

        seq = self.self.get_PCA_seq(input, use_kernel = True, kernel = 'polynomial')
        # plot_seq_continuous(seq, show = True, save = False, title = 'kPCA test')

        seq = self.self.get_PCA_seq(input, normalize = True)
        # plot_seq_continuous(seq, show = True, save = False, title = 'PCA-normalize test')

        seq = self.self.get_RPCA_seq(y, normalize = True, max_it = 500)
        plot_seq_continuous(seq, show = True, save = False, title = 'RPCA test')


    def test_suite(self):
        self.test_nmf_k_means()
        # self.test_random()
        # self.test_epi()
        # self.test_ChromHMM()
        # self.test_GNN()
        # self.test_PCA()


if __name__ ==  "__main__":
    main()
    # Tester().test_suite()
