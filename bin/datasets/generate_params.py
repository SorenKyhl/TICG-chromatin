import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import multivariate_normal, norm, qmc, skewnorm

sys.path.insert(0, '/home/erschultz/TICG-chromatin')
from scripts.ECDF import Ecdf
from scripts.get_params import GetSeq

sys.path.insert(0, '/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y
from sequences_to_contact_maps.scripts.utils import triu_to_full


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
        self.chi_param_version = args.chi_param_version

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
                    with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k4_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                chi_ij = skewnorm.rvs(-1.722, 6.908, 12.136, size = int(self.k*(self.k-1)/2))
            elif self.chi_param_version == 'v8':
                chi_ii = np.zeros(self.k)
                for j, l in enumerate('ABCD'):
                    with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k4_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                chi_ij = []
                for j, l1 in enumerate('ABCD'):
                    for k, l2 in enumerate('ABCD'):
                        if k > j:
                            with open(osp.join(f'/home/erschultz/dataset_01_26_23/plaid_param_distributions/k4_chi{l1}{l2}.pickle'), 'rb') as f:
                                dict_j = pickle.load(f)
                            chi_ij.append(skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma']))

            if self.chi_param_version not in {'v4', 'v5'}:
                chi = np.zeros((self.k, self.k))
                np.fill_diagonal(chi, chi_ii)
                chi[np.triu_indices(self.k, 1)] = chi_ij
                chi = chi + np.triu(chi, 1).T

            chi_file = osp.join(self.odir, f'chi_{i+1}.npy')
            np.save(chi_file, chi)

            chi_file = osp.join('/home/erschultz', self.dataset, f'setup/chi_{i+1}.npy')
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
            self.sample_folder = osp.join(dataset, f'sample{j}')
            _, y_diag = load_Y(self.sample_folder)
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

    def linear_params(self):
        if self.m == 512:
            with open(osp.join(self.dir, 'dataset_01_26_23/diagonal_param_distributions/k4_linear_intercept.pickle'), 'rb') as f:
                dict_intercept = pickle.load(f)
            with open(osp.join(self.dir, 'dataset_01_26_23/diagonal_param_distributions/k4_linear_slope.pickle'), 'rb') as f:
                dict_slope = pickle.load(f)

            for i in range(self.N):
                self.sample_dict[i]['diag_chi_method'] = 'linear'
                self.sample_dict[i]['diag_chi_slope'] = skewnorm.rvs(dict_slope['alpha'], dict_slope['mu'], dict_slope['sigma'])
                self.sample_dict[i]['diag_chi_constant'] = skewnorm.rvs(dict_intercept['alpha'], dict_intercept['mu'], dict_intercept['sigma'])

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

    def get_dataset(self):
        if self.diag_mode == 'linear':
            self.linear_params()
        elif self.diag_mode == 'logistic':
            self.logistic_params()

        self.plaid_params()

        if self.seq_mode == 'markov':
            self.seq_markov_params()
        elif self.seq_mode == 'pcs':
            self.seq_pc_params()
        elif self.seq_mode == 'pcs_shuffle':
            self.seq_pc_params(True)


        # write to odir
        for i in range(self.N):
            ofile = osp.join(self.odir, f'sample_{i+1}.txt')
            with open(ofile, 'w') as f:
                for key, val in self.sample_dict[i].items():
                    f.write(f'--{key}\n{val}\n')


def main():
    print(os.getcwd())
    args = getArgs()
    generator = DatasetGenerator(args)
    generator.get_dataset()

if __name__ == '__main__':
    main()
