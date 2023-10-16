import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import numpy as np
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_diag_chi_step, calculate_L
from pylib.utils.utils import pearson_round, triu_to_full
from scipy.stats import laplace, multivariate_normal, norm, qmc, skewnorm
from sklearn.neighbors import KernelDensity

sys.path.insert(0, '/home/erschultz/TICG-chromatin')
from scripts.data_generation.ECDF import Ecdf
from scripts.get_params_old import GetSeq

sys.path.insert(0, '/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (load_import_log,
                                                          load_max_ent_S,
                                                          load_Y)

LETTERS='ABCDEFGHIJKLMN'

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser', allow_abbrev = False)

    # input data args
    parser.add_argument('--dataset', type=str,
                        help='output dataset')
    parser.add_argument('--k', type=int,
                        help='number of marks in max ent')
    parser.add_argument('--sim_k', type=int,
                        help='number of marks in simulation')
    parser.add_argument('--ar', type=float,
                        help='aspect ratio for spheroid boundary')
    parser.add_argument('--samples', type=int,
                        help='number of samples')
    parser.add_argument('--m', type=int,
                        help='number of beads')
    parser.add_argument('--diag_mode', type=str, default='logistic',
                    help='mode for diagonal parameters')
    parser.add_argument('--seq_mode', type=str, default='markov',
                    help='mode for seq parameters')
    parser.add_argument('--chi_param_version', type=str,
                    help='version for chi param distribution (deprecated)')
    parser.add_argument('--max_L', type=float,
                    help='Any S_ij > max will be cropped to max')
    parser.add_argument('--grid_mode', type=str,
                    help='How to determine grid size')
    parser.add_argument('--data_dir', type=str, default='/project/depablo/erschultz',
                    help='where data will be found when running simulation')
    parser.add_argument('--exp_dataset', type=str, default='dataset_02_04_23',
                    help='dataset where experimental data is located')
    parser.add_argument('--b', type=int, default=140,
                        help='bond length')
    parser.add_argument('--phi', type=float,
                    help='phi chromatin')
    parser.add_argument('--v', type=int,
                    help='simulation volume')
    parser.add_argument('--conv_defn', type=str, default='loss')
    parser.add_argument('--plaid_mode', type=str, default='skewnorm')



    args = parser.parse_args()
    return args


class DatasetGenerator():
    def __init__(self, args):
        self.N = args.samples
        self.m = args.m
        self.k = args.k
        if args.sim_k is None:
            self.sim_k = args.k
        else:
            self.sim_k = args.sim_k
        self.dataset = args.dataset
        self.diag_mode = args.diag_mode
        self.seq_mode = args.seq_mode
        self.grid_mode = args.grid_mode
        self.chi_param_version = args.chi_param_version
        self.max_L = args.max_L
        self.data_dir = args.data_dir
        self.exp_dataset = args.exp_dataset
        self.b = args.b
        self.phi = args.phi
        self.v = args.v
        self.conv_defn = args.conv_defn
        self.plaid_mode = args.plaid_mode

        if self.phi is not None:
            assert self.v is None
            self.grid_root = f'optimize_grid_b_{self.b}_phi_{self.phi}'
            self.distributions_root = f'b_{self.b}_phi_{self.phi}_distributions'
        else:
            self.grid_root = f'optimize_grid_b_{self.b}_v_{self.v}'
            self.distributions_root = f'b_{self.b}_v_{self.v}_distributions'

        if args.ar != 1:
            self.grid_root += f'_spheroid_{args.ar}'
            self.distributions_root += f'_spheroid_{args.ar}_distributions'

        self.get_exp_samples()

        self.dir = '/home/erschultz'
        odir = osp.join(self.dir, self.dataset)
        if not osp.exists(odir):
            os.mkdir(odir, mode = 0o755)
        self.odir = osp.join(odir, 'setup')
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)

        self.exp_dir =  osp.join(self.dir, self.exp_dataset, 'samples')

        # sample : dictionary of params
        self.sample_dict = defaultdict(dict)
        for i in range(self.N):
            self.sample_dict[i]['m'] = self.m
            if args.ar != 1:
                self.sample_dict[i]['boundary_type'] = 'spheroid'
                self.sample_dict[i]['aspect_ratio'] = args.ar


    def get_exp_samples(self):
        if self.exp_dataset == 'dataset_02_04_23':
            exp_samples = range(201, 283)
            assert self.m == 512, f"m={self.m}"
        else:
            raise Exception(f'unrecognized dataset {self.exp_dataset}')

        # only use odd samples
        odd_samples = []
        for s in exp_samples:
            s_dir = osp.join('/home/erschultz', self.exp_dataset, f'samples/sample{s}')
            result = load_import_log(s_dir)
            chrom = int(result['chrom'])
            if chrom % 2 == 1:
                odd_samples.append(s)
        self.exp_samples = odd_samples

    def plaid_params(self):
        if self.plaid_mode == 'none':
            for i in range(self.N):
                self.sample_dict[i]['k'] = 0
                self.sample_dict[i]['chi_method'] = 'none'
            return
        for i in range(self.N):
            self.sample_dict[i]['k'] = self.sim_k
            if self.k == 0:
                self.sample_dict[i]['chi_method'] = 'none'
                continue

            # eignorm approach
            chi_ii = np.zeros(self.sim_k)
            for j in range(self.sim_k):
                l = LETTERS[j]
                if self.plaid_mode == 'skewnorm':
                    with open(osp.join(self.dir, self.exp_dataset, self.distributions_root,
                                        'plaid_param_distributions_eig_norm',
                                        f'k{self.k}_chi{l}{l}.pickle'), 'rb') as f:
                        dict_j = pickle.load(f)
                    chi_ii[j] =  skewnorm.rvs(dict_j['alpha'], dict_j['mu'], dict_j['sigma'])
                elif self.plaid_mode == 'KDE':
                    with open(osp.join(self.dir, self.exp_dataset, self.distributions_root,
                                        'plaid_param_distributions_eig_norm',
                                        f'k{self.k}_chi{l}{l}_KDE.pickle'), 'rb') as f:
                        kde = pickle.load(f)
                    chi_ii[j] = kde.sample(1).reshape(-1)
            chi_ij = np.zeros(int(self.sim_k*(self.sim_k-1)/2))

            chi = np.zeros((self.sim_k, self.sim_k))
            np.fill_diagonal(chi, chi_ii)
            chi[np.triu_indices(self.sim_k, 1)] = chi_ij
            chi = chi + np.triu(chi, 1).T

            chi_file = osp.join(self.odir, f'chi_{i+1}.npy')
            np.save(chi_file, chi)

            chi_file = osp.join(self.data_dir, self.dataset, f'setup/chi_{i+1}.npy')
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

        for j in self.exp_samples:
            sample_folder = osp.join(self.exp_dir, f'sample{j}')
            _, y_diag = load_Y(sample_folder)
            getseq = GetSeq(self, None, False)
            x = getseq.get_PCA_seq(y_diag, normalize = True)
            x_dict[j] = x


        for i in range(self.N):
            j = np.random.choice(self.exp_samples)
            x = x_dict[j]
            if shuffle:
                np.random.shuffle(x)

            seq_file = osp.join(self.odir, f'x_{i+1}.npy')
            np.save(seq_file, x)

            seq_file = osp.join(self.data_dir, self.dataset, f'setup/x_{i+1}.npy')
            self.sample_dict[i]['method'] = seq_file

    def seq_eig_params(self):
        x_dict = {} # id : x
        for j in self.exp_samples:
            sample_folder = osp.join(self.exp_dir,  f'sample{j}')
            max_ent_folder = osp.join(sample_folder, f'{self.grid_root}-max_ent{self.k}/resources')
            x = np.load(osp.join(max_ent_folder, 'x_eig_norm.npy'))
            x_dict[j] = x

        for i in range(self.N):
            j = np.random.choice(self.exp_samples)
            x = x_dict[j]
            x = x[:, :self.sim_k]

            seq_file = osp.join(self.odir, f'x_{i+1}.npy')
            np.save(seq_file, x)

            seq_file = osp.join(self.data_dir, self.dataset, f'setup/x_{i+1}.npy')
            self.sample_dict[i]['method'] = seq_file

    def linear_params(self):
        if self.m == 512:
            dir = osp.join(self.dir, self.exp_dataset, 'diagonal_param_distributions')

            if self.k == 0:
                k = 8
            else:
                k = self.k
            with open(osp.join(dir, f'k{k}_linear_intercept.pickle'), 'rb') as f:
                dict_intercept = pickle.load(f)
            with open(osp.join(dir, f'k{k}_linear_slope.pickle'), 'rb') as f:
                dict_slope = pickle.load(f)

            for i in range(self.N):
                self.sample_dict[i]['diag_chi_method'] = 'linear'

                slope = skewnorm.rvs(dict_slope['alpha'], dict_slope['mu'], dict_slope['sigma'])
                self.sample_dict[i]['diag_chi_slope'] = slope * 1000

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

    def get_converged_samples(self):
        converged_samples = []
        for i in self.exp_samples:
            sample_folder = osp.join(self.exp_dir, f'sample{i}', f'{self.grid_root}-max_ent{self.k}')
            converged = False

            # check convergence
            if self.conv_defn == 'loss':
                convergence_file = osp.join(sample_folder, 'convergence.txt')
                eps = 1e-2
                if osp.exists(convergence_file):
                    conv = np.loadtxt(convergence_file)
                    for j in range(1, len(conv)):
                        diff = conv[j] - conv[j-1]
                        if np.abs(diff) < eps and conv[j] < conv[0]:
                            converged = True
                else:
                    print(f'Warning: {convergence_file} does not exist')
            elif self.conv_defn == 'param':
                all_chis = []
                all_diag_chis = []
                for j in range(31):
                    it_path = osp.join(sample_folder, f'iteration{j}')
                    if osp.exists(it_path):
                        config_file = osp.join(it_path, 'production_out/config.json')
                        with open(config_file) as f:
                            config = json.load(f)
                        chis = np.array(config['chis'])
                        chis = chis[np.triu_indices(len(chis))] # grab upper triangle
                        diag_chis = np.array(config['diag_chis'])

                        all_chis.append(chis)
                        all_diag_chis.append(diag_chis)

                params = np.concatenate((all_diag_chis, all_chis), axis = 1)

                convergence = []
                eps = 1e2
                for j in range(5, len(params)):
                    diff = params[j] - params[j-1]
                    diff = np.linalg.norm(diff, ord = 2)
                    if diff < eps:
                        converged = True


            if converged:
                converged_samples.append(i)
            else:
                print(f'sample{i} did not converge')

        print('converged %:', len(converged_samples) / len(self.exp_samples) * 100)


        return converged_samples

    def max_ent_params(self):
        diag_dict = {} # id : diag_params
        grid_dict = {} # id : grid_size
        get_grid = False
        linear = False
        poly3 = False; poly6_log = False
        if 'grid' in self.diag_mode:
            get_grid = True
        if 'linear' in self.diag_mode:
            linear = True
        elif 'poly3' in self.diag_mode:
            poly3 = True
        elif 'poly6_log' in self.diag_mode:
            poly6_log = True

        converged_samples = self.get_converged_samples()
        print(converged_samples, len(converged_samples))
        for j in converged_samples:
            sample_folder = osp.join(self.exp_dir, f'sample{j}', f'{self.grid_root}-max_ent{self.k}')

            if linear:
                diag_chi_step = np.loadtxt(osp.join(sample_folder, 'fitting/linear_fit.txt'))
            elif poly3:
                diag_chi_step = np.loadtxt(osp.join(sample_folder, 'fitting/poly3_fit.txt'))
            elif poly6_log:
                diag_chi_step = np.loadtxt(osp.join(sample_folder, 'fitting2/poly6_log_fit.txt'))
            else:
                diag_chis = np.loadtxt(osp.join(sample_folder, 'chis_diag.txt'))
                with open(osp.join(sample_folder, 'resources/config.json'), 'r') as f:
                    config = json.load(f)
                diag_chi_step = calculate_diag_chi_step(config, diag_chis)
            diag_dict[j] = diag_chi_step

            # get grid_size
            if get_grid:
                grid_file = osp.join(self.exp_dir, f'sample{j}', f'{self.grid_root}/grid.txt')
                grid_dict[j] = np.loadtxt(grid_file)

        for i in range(self.N):
            j = np.random.choice(converged_samples)
            diag_chis = diag_dict[j]

            diag_file = osp.join(self.odir, f'diag_chis_{i+1}.npy')
            np.save(diag_file, diag_chis)

            diag_file = osp.join(self.data_dir, self.dataset, f'setup/diag_chis_{i+1}.npy')

            self.sample_dict[i]['diag_chi_experiment'] = osp.join(self.exp_dataset,
                                                                f'samples/sample{j}',
                                                                f'{self.grid_root}')
            self.sample_dict[i]['diag_chi_method'] = diag_file
            self.sample_dict[i]['diag_bins'] = self.m

            if get_grid:
                self.sample_dict[i]['grid_size'] = grid_dict[j]


    def meanDist_S_params(self):
        meanDist_S_dict = {} # id : meanDist_S
        grid_dict = {} # id : grid_size
        get_grid = False
        if 'grid' in self.diag_mode:
            get_grid = True
        poly12 = False
        if 'poly12' in self.diag_mode:
            poly12 = True
            print('Using poly12 for meanDist_S')

        converged_samples = self.get_converged_samples()
        for j in converged_samples:
            sample_folder = osp.join(self.exp_dir, f'sample{j}', f'{self.grid_root}-max_ent{self.k}')
            if poly12:
                meanDist_S = np.loadtxt(osp.join(sample_folder, 'fitting2/poly12_log_meanDist_S_fit.txt'))
            else:
                meanDist_S = np.loadtxt(osp.join(sample_folder, 'fitting2/poly6_log_meanDist_S_fit.txt'))
            meanDist_S_dict[j] = meanDist_S

            # get grid_size
            if get_grid:
                grid_dict[j] = np.loadtxt(osp.join(self.exp_dir, f'sample{j}', f'{self.grid_root}/grid.txt'))

        for i in range(self.N):
            j = np.random.choice(converged_samples)
            meanDist_S = meanDist_S_dict[j]

            # adjust based on L
            if 'method' in self.sample_dict[i]:
                file = osp.split(self.sample_dict[i]['method'])[-1]
                path = osp.join(f'/home/erschultz/{self.dataset}/setup/', file)
                seq = np.load(path)
                file = osp.split(self.sample_dict[i]['chi_method'])[-1]
                path = osp.join(f'/home/erschultz/{self.dataset}/setup/', file)
                chi = np.load(path)
                L = calculate_L(seq, chi)

                meanDist_L = DiagonalPreprocessing.genomic_distance_statistics(L, 'freq')
                diag_chis = meanDist_S - meanDist_L
            else:
                diag_chis = meanDist_S

            diag_file = osp.join(self.odir, f'diag_chis_{i+1}.npy')
            np.save(diag_file, diag_chis)
            diag_file = osp.join(self.data_dir, self.dataset, f'setup/diag_chis_{i+1}.npy')

            self.sample_dict[i]['diag_chi_experiment'] = osp.join(self.exp_dataset,
                                                                f'samples/sample{j}',
                                                                f'{self.grid_root}')
            self.sample_dict[i]['diag_chi_method'] = diag_file
            self.sample_dict[i]['diag_bins'] = self.m


            if get_grid:
                self.sample_dict[i]['grid_size'] = grid_dict[j]


    def grid_params(self):
        with open(osp.join(self.dir, self.exp_dataset,
                            self.distributions_root,
                            'grid_size.pickle'), 'rb') as f:
            dict_grid = pickle.load(f)

        for i in range(self.N):
            grid_size = skewnorm.rvs(dict_grid['alpha'], dict_grid['mu'], dict_grid['sigma'])
            self.sample_dict[i]['grid_size'] = grid_size

    def get_dataset(self):
        if self.seq_mode == 'markov':
            self.seq_markov_params()
        elif self.seq_mode == 'pcs':
            self.seq_pc_params()
        elif self.seq_mode == 'norm':
            self.seq_eig_params()
        elif self.seq_mode.startswith('eig_norm'):
            self.seq_eig_params()
        elif self.seq_mode == 'pcs_shuffle':
            self.seq_pc_params(True)

        self.plaid_params()

        if self.diag_mode.startswith('linear'):
            self.linear_params()
        elif self.diag_mode == 'logistic':
            self.logistic_params()
        elif self.diag_mode.startswith('max_ent'):
            self.max_ent_params()
        elif self.diag_mode.startswith('meanDist_S'):
            self.meanDist_S_params()

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
        print(f'Writing to {self.odir}')
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
