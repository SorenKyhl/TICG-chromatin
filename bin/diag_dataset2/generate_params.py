import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
from scipy.stats import norm, qmc, skewnorm


def linear_dataset(N, k, dataset):
    data_dir = osp.join('/home/erschultz', dataset)
    if not osp.exists(data_dir):
        os.mkdir(data_dir, mode = 0o755)
    odir = osp.join(data_dir, 'setup')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    sample_dict = defaultdict(dict) # sample : dictionary of params

    # first get diagonal params
    l_bounds = [0.001, 0]
    u_bounds = [0.004, 8]
    sampler = qmc.LatinHypercube(d=2) # slope, intercept
    sample = sampler.random(N)
    sample = qmc.scale(sample, l_bounds, u_bounds)

    for i, vals in enumerate(sample):
        sample_dict[i]['diag_chi_method'] = 'linear'
        sample_dict[i]['diag_chi_slope'] = vals[0] * 1000
        sample_dict[i]['diag_chi_constant'] = vals[1]

    # get plaid chis
    for i in range(N):
        sample_dict[i]['k'] = k
        chi_ii = skewnorm.rvs(-1.549, 0.742, 3.022, size = k)
        chi_ij = skewnorm.rvs(-1.85, 2.132, 3.465, size = int(k*(k-1)/2))
        chi = np.zeros((k,k))
        np.fill_diagonal(chi, chi_ii)
        chi[np.triu_indices(k, 1)] = chi_ij
        chi = chi + np.triu(chi, 1).T

        chi_file = osp.join(odir, f'chi_{i+1}.npy')
        np.save(chi_file, chi)

        chi_file = osp.join('/project2/depablo/erschultz', dataset, f'setup/chi_{i+1}.npy')
        sample_dict[i]['chi_method'] = chi_file

    # get seq params
    for i in range(N):
        lmbda = skewnorm.rvs(0, 0.865, 0.06)
        f = skewnorm.rvs(-1.091, 0.426, 0.15)
        sample_dict[i]['lmbda'] = lmbda
        sample_dict[i]['f'] = f

    for i in range(N):
        ofile = osp.join(odir, f'sample_{i+1}.txt')
        with open(ofile, 'w') as f:
            for key, val in sample_dict[i].items():
                f.write(f'--{key}\n{val}\n')

def logistic_dataset(N):
    pass

def main():
    linear_dataset(2500, 8, 'dataset_11_18_22')
    logistic_dataset(2500)

if __name__ == '__main__':
    main()
