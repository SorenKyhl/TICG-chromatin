import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
from scipy.stats import norm, qmc, skewnorm


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser', allow_abbrev = False)

    # input data args
    parser.add_argument('--dataset', type=str,
                        help='output dataset')
    parser.add_argument('--k', type=int,
                        help='number of marks')
    parser.add_argument('--samples', type=int,
                        help='number of samples')
    parser.add_argument('--diag_mode', type=str, default='logistic',
                    help='mode for diagonal parameters')
    parser.add_argument('--seq_mode', type=str, default='markov',
                    help='mode for seq parameters')

    args = parser.parse_args()
    return args


class DatasetGenerator():
    def __init__(self, N, k, dataset, diag_mode, seq_mode):
        self.N = N
        self.k = k
        self.dataset = dataset
        self.diag_mode = diag_mode
        self.seq_mode = seq_mode

        data_dir = osp.join('/home/erschultz', dataset)
        if not osp.exists(data_dir):
            os.mkdir(data_dir, mode = 0o755)
        self.odir = osp.join(data_dir, 'setup')
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)

        # sample : dictionary of params
        self.sample_dict = defaultdict(dict)

    def plaid_params(self):
        for i in range(self.N):
            self.sample_dict[i]['k'] = self.k
            chi_ii = skewnorm.rvs(0.83, -2.459, 2.594, size = self.k)
            chi_ij = skewnorm.rvs(-1.566, 1.815, 2.892, size = int(self.k*(self.k-1)/2))
            chi = np.zeros((self.k, self.k))
            np.fill_diagonal(chi, chi_ii)
            chi[np.triu_indices(self.k, 1)] = chi_ij
            chi = chi + np.triu(chi, 1).T

            chi_file = osp.join(self.odir, f'chi_{i+1}.npy')
            np.save(chi_file, chi)

            chi_file = osp.join('/project2/depablo/erschultz', self.dataset, f'setup/chi_{i+1}.npy')
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


    def seq_pc_params(self):
        for i in range(self.N):
            j = np.random.choice(range(2201, 2216))
            x = np.load(f'/home/erschultz/dataset_11_14_22/samples/sample{j}/PCA-normalize-E/k8/replicate1/resources/x.npy')

            seq_file = osp.join(self.odir, f'x_{i+1}.npy')
            np.save(seq_file, x)

            seq_file = osp.join('/project2/depablo/erschultz', self.dataset, f'setup/x_{i+1}.npy')
            self.sample_dict[i]['method'] = seq_file

    def linear_params(self):
        # first get diagonal params
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

        # write to odir
        for i in range(self.N):
            ofile = osp.join(self.odir, f'sample_{i+1}.txt')
            with open(ofile, 'w') as f:
                for key, val in self.sample_dict[i].items():
                    f.write(f'--{key}\n{val}\n')


def main():
    args = getArgs()
    generator = DatasetGenerator(args.samples, args.k, args.dataset,
                                args.diag_mode, args.seq_mode)
    generator.get_dataset()

if __name__ == '__main__':
    main()
