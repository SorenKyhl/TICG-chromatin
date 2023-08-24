import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from modify_maxent import get_samples
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_diag_chi_step

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y


def split_samples(dataset):
    samples, _ = get_samples(dataset)
    good = []; bad = []
    for s in samples:
        s_dir = osp.join('/home/erschultz', dataset, f'samples/sample{s}')
        y, _ = load_Y(s_dir)
        y /= np.mean(np.diagonal(y))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
        if meanDist[10] > 0.06:
            bad.append(s)
        else:
            good.append(s)

    return good, bad

def compare_diag_params():
    dataset = 'dataset_08_17_23'
    data_dir = osp.join('/home/erschultz', dataset)
    good, bad = split_samples(dataset)
    print(f'good = {len(good)}, bad = {len(bad)}')

    def get_data(s):
        s_dir = osp.join(data_dir, f'samples/sample{s}')
        diag_chis = np.load(osp.join(s_dir, 'diag_chis.npy'))
        chis = np.load(osp.join(s_dir, 'chis.npy'))
        chis = chis.diagonal()
        with open(osp.join(s_dir, 'config.json')) as f:
            config = json.load(f)
        setup_file = osp.join(data_dir, f'setup/sample_{s}.txt')
        with open(setup_file) as f:
            for line in f:
                if line.startswith('--diag_chi_experiment'):
                    line = f.readline()
                    exp = line.strip()

        exp_chis = np.load(osp.join('/home/erschultz', exp+'-max_ent10/chis_eig_norm.npy'))
        exp_chis = exp_chis.diagonal()
        return diag_chis, chis, exp_chis, config

    for i in good:
        diag_chis, chis, exp_chis, config = get_data(i)
        plt.plot(chis, c='g')

    for i in bad:
        diag_chis, chis, exp_chis, config = get_data(i)
        plt.plot(chis, c='r')
    plt.xlabel('Chi')
    plt.ylabel('Value')
    # plt.xscale('log')
    plt.savefig('/home/erschultz/TICG-chromatin/figures/bad_vs_good_chis.png')
    plt.close()




if __name__ == '__main__':
    compare_diag_params()
