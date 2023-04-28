import json
import os
import os.path as osp
import sys

import numpy as np

from pylib.pysim import Pysim

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.plotting_utils import plot_matrix


def sim_downsample_dataset(dataset, sample):
    dir = '/home/erschultz'
    sample_dir = osp.join(dir, dataset, f'samples/sample{sample}')
    odir = osp.join(dir, 'dataset_downsampling')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    odir = osp.join(odir, 'samples')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    for i in range(1, 6):
        sweeps = i * 100000
        y = np.loadtxt(osp.join(sample_dir, f'data_out/contacts{sweeps}.txt'))
        y /= np.mean(np.diagonal(y))

        odir_sample = osp.join(odir, f'sample{i}')
        if not osp.exists(odir_sample):
            os.mkdir(odir_sample, mode=0o755)
        np.save(osp.join(odir_sample, 'y.npy'), y)
        plot_matrix(y, osp.join(odir_sample, 'y.png'), vmax='mean')




def main():
    sim_downsample_dataset('dataset_03_22_23', 324)

if __name__ == '__main__':
    main()
