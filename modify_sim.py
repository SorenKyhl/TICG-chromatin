import json
import multiprocessing as mp
import os
import os.path as osp
import sys

import numpy as np
import optimize_grid
import pylib.analysis as analysis
import scipy
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_all_energy
from scripts.get_params import GetSeq


def smatrix_mode(config):
    config['chis'] = None
    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['diag_chis'] = None
    config['diagonal_on'] = False
    config['dmatrix_on'] = False
    config['lmatrix_on'] = False
    config['smatrix_filename'] = 'smatrix.txt'
    return config

def only_diag_mode(config):
    config['chis'] = None
    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['dmatrix_on'] = True
    config['lmatrix_on'] = False
    config['smatrix_on'] = False
    return config


def modify(sample):
    dataset = 'dataset_02_04_23'
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}/optimize_grid_b_140_phi_0.03-max_ent'
    y = np.load(osp.join(dir, 'y.npy'))
    y /= np.mean(np.diagonal(y))
    nbeads = len(y)

    root = osp.join(dir, 'copy_S')
    print(root)
    config = utils.load_json(osp.join(dir, 'config.json'))
    config['bead_type_files'] = [f'pcf{i}.txt' for i in range(1, config['nspecies']+1)]
    config['track_contactmap'] = False

    # get sequences
    seqs = np.load(osp.join(dir, 'x.npy'))

    L, D, S = calculate_all_energy(config, seqs, np.array(config["chis"]))
    # meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
    # S = scipy.linalg.toeplitz(meanDist)
    config = smatrix_mode(config)
    print(D)

    # sim = Pysim(root, config, None, y, randomize_seed = False, overwrite = True, smatrix = S)
    # sim.run_eq(10000, 50000, 5)
    #
    # with utils.cd(sim.root):
    #     analysis.main_no_maxent()


def main():
    # with mp.Pool(17) as p:
        # p.map(fit, range(202, 283))
    # for i in range(202, 283):
        # fit(i)
    modify(203)


if __name__ == '__main__':
    # soren()
    # modify_soren()
    main()
