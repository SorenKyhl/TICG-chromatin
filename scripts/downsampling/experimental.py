import json
import os
import os.path as osp
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from pylib.utils.plotting_utils import plot_matrix

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.utils import triu_to_full


def make_samples():
    exponents = np.arange(4, 9)
    dir = '/home/erschultz/downsampling_analysis'
    for exponent in exponents:
        e_dir = f'{dir}/samples_exp{exponent}'
        if not osp.exists(e_dir):
            os.mkdir(e_dir, mode=0o755)

    for s_exp in range(201, 211):
        exp_dir = f'/home/erschultz/dataset_02_04_23/samples/sample{s_exp}'
        y = np.triu(np.load(osp.join(exp_dir, 'y.npy')))
        p = y / np.sum(y)
        m = len(y)
        p_flat = p[np.triu_indices(m)]
        pos = np.arange(0, len(p_flat))

        for exponent in exponents:
            e_dir = f'{dir}/samples_exp{exponent}'
            print(exponent)
            odir = osp.join(e_dir, f'sample{s_exp}')
            os.mkdir(odir, mode = 0o755)
            count = 10**exponent
            choices = np.random.choice(pos, size = count, p = p_flat)
            y_i_flat = np.zeros_like(p_flat)
            for j in choices:
                # this is really slow
                y_i_flat[j] += 1
            print(np.sum(y_i_flat))
            y_i = triu_to_full(y_i_flat)
            np.save(osp.join(odir, 'y.npy'), y_i)
            plot_matrix(y_i, osp.join(odir, 'y.png'), vmax = 'mean')




if __name__ == '__main__':
    make_samples()
