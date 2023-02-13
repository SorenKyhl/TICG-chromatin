import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import laplace, multivariate_normal, norm, qmc, skewnorm

sys.path.insert(0, '/home/erschultz/TICG-chromatin')
from scripts.ECDF import Ecdf
from scripts.get_params import GetSeq

sys.path.insert(0, '/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y
from sequences_to_contact_maps.scripts.utils import pearson_round, triu_to_full


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser', allow_abbrev = False)

    # input data args
    parser.add_argument('--dataset', type=str,
                        help='output dataset')
    parser.add_argument('--k', type=int,
                        help='number of marks')
    parser.add_argument('--samples', type=int,
                        help='number of samples')
    parser.add_argument('--m', type=int,
                        help='number of beads')
    parser.add_argument('--diag_mode', type=str, default='logistic',
                    help='mode for diagonal parameters')
    parser.add_argument('--seq_mode', type=str, default='markov',
                    help='mode for seq parameters')
    parser.add_argument('--chi_param_version', type=str, default='v1',
                    help='version for chi param distribution')
    parser.add_argument('--max_L', type=float,
                    help='Any S_ij > max will be cropped to max')
    parser.add_argument('--grid_mode', type=str,
                    help='How to determine grid size')

    args = parser.parse_args()
    return args


class DatasetGenerator():
    def __init__(self, args):
        self.N = args.samples
        self.m = args.m
        self.k = args.k
        self.dataset = args.dataset
        self.diag_mode = args.diag_mode
        self.seq_mode = args.seq_mode
        self.grid_mode = args.grid_mode
        self.chi_param_version = args.chi_param_version
        self.max_L = args.max_L

        self.dir = '/home/erschultz'
        data_dir = osp.join(self.dir, self.dataset)
        if not osp.exists(data_dir):
            os.mkdir(data_dir, mode = 0o755)
        self.odir = osp.join(data_dir, 'setup')
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)

        # sample : dictionary of params
        self.sample_dict = defaultdict(dict)

    def plaid_params(self):
        if self.chi_param_version == 'v3':
            dist_ii = Ecdf(fname = osp.join('/home/erschultz/dataset_11_14_22/k8_chi_ii.json'))
            dist_ij = Ecdf(fname = osp.join('/home/erschultz/dataset_11_14_22/k8_chi_ij.json'))
        elif self.chi_param_version == 'v4':
            with open(osp.join(self.dir, 'dataset_11_14_22/plaid_param_distributions/k4_chi_multivariate.pickle'), 'rb') as f:
                dict = pickle.load(f)
                cov = dict['cov']
                mean = dict['mean']
            dist = multivariate_normal(mean, cov)
        elif self.chi_param_version == 'v5':
            with open(osp.join('/home/erschultz/dataset_01_26_23/plaid_param_distributions/k4_chi_multivariate.pickle'), 'rb') as f:
                dict = pickle.load(f)
                cov = dict['cov']
                mean = dict['mean']
            dist = multivariate_normal(mean, cov)


        for i in range(self.N):
            self.sample_dict[i]['k'] = self.k
            if self.chi_param_version == 'v1':
                chi_ii = skewnorm.rvs(0.83, -2.459, 2.594, size = self.k)
                chi_ij = skewnorm.rvs(-1.566, 1.815, 2.892, size = int(self.k*(self.k-1)/2))
            elif self.chi_param_version == 'v2':
                chi_ii = skewnorm.rvs(-3.619, 2.119, 5.244, size = self.k)
                chi_ij = skewnorm.rvs(-1.307, 3.257, 5.888, size = int(self.k*(self.k-1)/2))
            elif self.chi_param_version == 'v3':
                chi_ii = dist_ii.rvs(size = self.k)
                chi_ij = dist_ij.rvs(size = int(self.k*(self.k-1)/2))
            elif self.chi_param_version in {'v4', 'v5'}:
                chi = dist.rvs()
                chi = triu_to_full(chi)
            elif self.chi_param_version == 'v6':
                chi_ii = skewnorm.rvs(-6.91, 1.739, 11.897, size = self.k)
                chi_ij = skewnorm.rvs(-1.722, 6.908, 12.136, size = int(self.k*(self.k-1)/2))
            elif self.chi_param_version == 'v7':
                chi_ii = np.zeros(self.k)
                for j, l in enumerate('ABCD'):
                    with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k{self.k}_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                chi_ij = skewnorm.rvs(-1.722, 6.908, 12.136, size = int(self.k*(self.k-1)/2))
            elif self.chi_param_version == 'v8':
                chi_ii = np.zeros(self.k)
                for j, l in enumerate('ABCD'):
                    with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k{self.k}_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                chi_ij = []
                for j, l1 in enumerate('ABCD'):
                    for k, l2 in enumerate('ABCD'):
                        if k > j:
                            with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k4_chi{l1}{l2}.pickle'), 'rb') as f:
                                dict_j = pickle.load(f)
                            chi_ij.append(skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma']))
            elif self.chi_param_version == 'v9':
                chi_ii = np.zeros(self.k)
                for j, l in enumerate('ABCD'):
                    with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k{self.k}_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                chi_ij = np.zeros(int(self.k*(self.k-1)/2))
            elif self.chi_param_version == 'v10':
                chi_ii = np.zeros(self.k)
                for j, l in enumerate('ABCD'):
                    with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k{self.k}_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                chi_ij = laplace.rvs(-0.109, 5.631, size = int(self.k*(self.k-1)/2))
            elif self.chi_param_version == 'v11':
                chi_ii = np.zeros(self.k)
                for j, l in enumerate('ABCD'):
                    with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k4_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])

                # grab x.npy to check corr(seq_A, seq_B)
                x = np.load(osp.join('/home/erschultz', self.dataset, f'setup/x_{i+1}.npy'))
                chi_ij = []
                for j, l1 in enumerate('ABCD'):
                    for k, l2 in enumerate('ABCD'):
                        if k > j:
                            corr = pearson_round(x[:, j], x[:, k])
                            val = np.abs(skewnorm.rvs(-1.722, 6.908, 12.136))
                            if corr < 0:
                                val *= -1
                            chi_ij.append(val)
            elif self.chi_param_version == 'v12':
                # eignorm approach
                chi_ii = np.zeros(self.k)
                for j, l in enumerate('ABCD'):
                    assert self.m == 512
                    with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions_eig_norm/k{self.k}_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                chi_ij = np.zeros(int(self.k*(self.k-1)/2))
            elif self.chi_param_version == 'v13':
                # eignorm approach
                chi_ii = np.zeros(self.k)
                for j, l in enumerate('ABCD'):
                    assert self.m == 512
                    with open(osp.join(f'/home/erschultz/dataset_02_04_23/plaid_param_distributions_eig_norm/k{self.k}_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                chi_ij = np.zeros(int(self.k*(self.k-1)/2))

            if self.chi_param_version not in {'v4', 'v5'}:
                chi = np.zeros((self.k, self.k))
                np.fill_diagonal(chi, chi_ii)
                chi[np.triu_indices(self.k, 1)] = chi_ij
                chi = chi + np.triu(chi, 1).T

            chi_file = osp.join(self.odir, f'chi_{i+1}.npy')
            np.save(chi_file, chi)

            chi_file = osp.join('/project/depablo/erschultz', self.dataset, f'setup/chi_{i+1}.npy')
            self.sample_dict[i]['chi_method'] = chi_file

    def seq_markov_params(self):
        for i in range(self.N):
            p11 = -1
            p00 = -1
            while p00 < 0 or p11 < 0 or p00 > 1 or p11 > 1:
                f = skewnorm.rvs(-1.598, 0.451, 0.17)
                lmbda = skewnorm.rvs(-2.252, 0.928, 0.084)
                p11 = f*(1-lmbda)+lmbda
                p00 = f*(lmbda-1) + 1

            self.sample_dict[i]['lmbda'] = lmbda
            self.sample_dict[i]['f'] = f

    def seq_pc_params(self, shuffle = False):
        x_dict = {} # id : x
        self.plot = False
        self.args_file = None
        self.sample = None
        if self.m == 1024:
            dataset = '/home/erschultz/dataset_11_14_22/samples'
            samples = range(2201, 2216)
        elif self.m == 512:
            dataset = f'/home/erschultz/dataset_01_26_23/samples'
            samples = range(201, 250)

        for j in samples:
            sample_folder = osp.join(dataset, f'sample{j}')
            _, y_diag = load_Y(sample_folder)
            getseq = GetSeq(self, None, False)
            x = getseq.get_PCA_seq(y_diag, normalize = True)
            x_dict[j] = x


        for i in range(self.N):
            j = np.random.choice(samples)
            x = x_dict[j]
            if shuffle:
                np.random.shuffle(x)

            seq_file = osp.join(self.odir, f'x_{i+1}.npy')
            np.save(seq_file, x)

            seq_file = osp.join('/home/erschultz', self.dataset, f'setup/x_{i+1}.npy')
            self.sample_dict[i]['method'] = seq_file

    def seq_eig_params(self, norm=False):
        x_dict = {} # id : x
        if self.m == 1024:
            dataset = '/home/erschultz/dataset_11_14_22/samples'
            samples = range(2201, 2216)
        elif self.m == 512:
            dataset = f'/home/erschultz/dataset_01_26_23/samples'
            samples = range(201, 250)

        for j in samples:
            sample_folder = osp.join(dataset, f'sample{j}')
            max_ent_folder = osp.join(sample_folder, f'PCA-normalize-E/k{self.k}/replicate1/resources')
            if norm:
                x = np.load(osp.join(max_ent_folder, 'x_eig_norm.npy'))
            else:
                x = np.load(osp.join(max_ent_folder, 'x_eig.npy'))
            x_dict[j] = x

        for i in range(self.N):
            j = np.random.choice(samples)
            x = x_dict[j]

            seq_file = osp.join(self.odir, f'x_{i+1}.npy')
            np.save(seq_file, x)

            seq_file = osp.join('/project/depablo/erschultz', self.dataset, f'setup/x_{i+1}.npy')
            self.sample_dict[i]['method'] = seq_file

    def linear_params(self):
        if self.m == 512:
            if self.diag_mode == 'linear_v2':
                dir = osp.join(self.dir, 'dataset_02_04_23/diagonal_param_distributions')
            else:
                dir = osp.join(self.dir, 'dataset_01_26_23/diagonal_param_distributions')

            with open(osp.join(dir, f'k{self.k}_linear_intercept.pickle'), 'rb') as f:
                dict_intercept = pickle.load(f)
            with open(osp.join(dir, f'k{self.k}_linear_slope.pickle'), 'rb') as f:
                dict_slope = pickle.load(f)

            for i in range(self.N):
                self.sample_dict[i]['diag_chi_method'] = 'linear'

                slope = skewnorm.rvs(dict_slope['alpha'], dict_slope['mu'], dict_slope['sigma'])
                self.sample_dict[i]['diag_chi_slope'] = slope

                intercept = skewnorm.rvs(dict_intercept['alpha'], dict_intercept['mu'], dict_intercept['sigma'])
                self.sample_dict[i]['diag_chi_constant'] = intercept

        else:
            # l_bounds = [0.001, 0]
            # u_bounds = [0.004, 8]
            l_bounds = [0.002, 0]
            u_bounds = [0.01, 8]
            sampler = qmc.LatinHypercube(d=2) # slope, intercept
            sample = sampler.random(self.N)
            sample = qmc.scale(sample, l_bounds, u_bounds)

            for i, vals in enumerate(sample):
                self.sample_dict[i]['diag_chi_method'] = 'linear'
                self.sample_dict[i]['diag_chi_slope'] = vals[0] * 1000
                self.sample_dict[i]['diag_chi_constant'] = vals[1]

    def logistic_params(self):
        # first get diagonal params
        l_bounds = [0.001, -100, 3]
        u_bounds = [0.004, 600, 15]
        sampler = qmc.LatinHypercube(d=3) # slope, midpoint, max
        sample = sampler.random(self.N)
        sample = qmc.scale(sample, l_bounds, u_bounds)

        for i, vals in enumerate(sample):
            self.sample_dict[i]['diag_chi_method'] = 'logistic'
            self.sample_dict[i]['diag_chi_slope'] = vals[0] * 1000
            self.sample_dict[i]['diag_chi_midpoint'] = vals[1]
            self.sample_dict[i]['max_diag_chi'] = vals[2]
            self.sample_dict[i]['min_diag_chi'] = 0

    def grid_params(self):
        for i in range(self.N):
            if self.grid_mode == 'v1':
                with open(osp.join(f'/home/erschultz/dataset_02_04_23/grid_size_distributions/grid_size.pickle'), 'rb') as f:
                    dict_grid = pickle.load(f)
                grid_size = skewnorm.rvs(dict_grid['alpha'], dict_grid['mu'], dict_grid['sigma'])
            self.sample_dict[i]['grid_size'] = grid_size

    def get_dataset(self):
        if self.diag_mode.startswith('linear'):
            self.linear_params()
        elif self.diag_mode == 'logistic':
            self.logistic_params()

        if self.seq_mode == 'markov':
            self.seq_markov_params()
        elif self.seq_mode == 'pcs':
            self.seq_pc_params()
        elif self.seq_mode == 'norm':
            self.seq_eig_params()
        elif self.seq_mode == 'eig_norm':
            self.seq_eig_params(True)
        elif self.seq_mode == 'pcs_shuffle':
            self.seq_pc_params(True)

        self.plaid_params()

        if self.max_L is not None:
            for i in range(self.N):
                x = np.load(osp.join('/home/erschultz', self.dataset, f'setup/x_{i+1}.npy'))
                chi = np.load(osp.join('/home/erschultz', self.dataset, f'setup/chi_{i+1}.npy'))
                L = x @ chi @ x.T
                L = (L + L.T) / 2
                L[L > self.max_L] = self.max_L
                L_file = osp.join('/home/erschultz', self.dataset, f'setup/L_{i+1}.npy')
                np.save(L_file, L)
                self.sample_dict[i]['method'] = L_file + '-L'
                self.sample_dict[i]['chi_method'] = None

        if self.grid_mode is not None:
            self.grid_params()


        # write to odir
        for i in range(self.N):
            ofile = osp.join(self.odir, f'sample_{i+1}.txt')
            with open(ofile, 'w') as f:
                for key, val in self.sample_dict[i].items():
                    f.write(f'--{key}\n{val}\n')


def main():
    args = getArgs()
    generator = DatasetGenerator(args)
    generator.get_dataset()

if __name__ == '__main__':
    main()
