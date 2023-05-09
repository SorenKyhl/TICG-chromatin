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
from pylib.utils.energy_utils import *
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


def setup():
    config = default.config
    m=512
    config['nbeads'] = m
    config['nspecies'] = 4
    config['seed'] = 12
    config['bead_type_files'] = [f'pcf{i}.txt' for i in range(1, config['nspecies']+1)]
    config['track_contactmap'] = False
    config['beadvol'] = 130000
    config['bond_length'] = 140
    config['phi_chromatin'] = 0.03
    config['grid_size'] = 210
    config['dump_frequency'] = 1000
    config['dump_stats_frequency'] = 1

    # set up diag chis
    config['diagonal_on'] = True
    config['dense_diagonal_on'] = True
    config['n_small_bins'] = 64
    config["n_big_bins"] = 16
    config["small_binsize"] = 1
    config["big_binsize"] = 28
    config['diag_chis'] = np.linspace(0, 10, config['n_small_bins']+config["n_big_bins"])
    print(config['diag_chis'])

    # get sequences
    rng = np.random.default_rng(12)
    seq = rng.choice((0,1), size=(m,4))
    print('psi')
    print(seq[0])

    chis = rng.normal(size=(4,4))*2
    print('chis')
    print(chis)
    config['chis'] = chis.tolist()

    return config, seq, chis

def baseline(config, seq, chis):
    root = '/home/erschultz/consistency_check/baseline'
    config['smatrix_on'] = False
    config['dmatrix_on'] = False
    config['lmatrix_on'] = False

    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(1000, 1000, 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_only_l(config, seq, chis):
    root = '/home/erschultz/consistency_check/baseline_only_l'
    config['smatrix_on'] = False
    config['dmatrix_on'] = False


    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(1000, 1000, 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_only_d(config, seq, chis):
    root = '/home/erschultz/consistency_check/baseline_only_d'
    config['smatrix_on'] = False
    config['lmatrix_on'] = False

    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(1000, 1000, 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_energy_on(config, seq, chis):
    root = '/home/erschultz/consistency_check/baseline_energy_on'
    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(1000, 1000, 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def smatrix_only(config, seq, chis):
    root = '/home/erschultz/consistency_check/smatrix_only'
    L, D, S = calculate_all_energy(config, seq, chis)
    print('L')
    print(L)
    L = seq @ chis @ seq.T
    print('L')
    print(L)
    L = (L+L.T)/2
    print('L_sym')
    print(L)
    # print(S)
    config = smatrix_mode(config)

    S_prime = convert_L_to_Lp(S)

    S_prime2 = np.loadtxt(osp.join(root, 'equilibration/smatrix_prime.txt'))

    print(np.max(S_prime - S_prime2))

    # sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, smatrix = S)
    # sim.run_eq(1000, 1000, 1)
    #
    # with utils.cd(sim.root):
    #   analysis.main_no_compare()

def smatrix_only_zeros(config, seq, chis):
    root = '/home/erschultz/consistency_check/smatrix_only_zeros'
    S = np.zeros((512, 512))
    # print(S)
    config = smatrix_mode(config)
    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, smatrix = S)
    sim.run_eq(1000, 1000, 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def lmatrix_dmatrix_only(config, seq, chis):
    root = '/home/erschultz/consistency_check/lmatrix_dmatrix_only'
    L, D, S = calculate_all_energy(config, seq, chis)
    config['chis'] = None
    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['dmatrix_on'] = True
    config['lmatrix_on'] = True
    config['smatrix_on'] = False
    config['lmatrix_filename'] = 'lmatrix.txt'
    config['dmatrix_filename'] = 'dmatrix.txt'


    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, lmatrix = L,
                dmatrix = D)
    sim.run_eq(1000, 1000, 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def lmatrix_dmatrix_only_use_S(config, seq, chis):
    root = '/home/erschultz/consistency_check/lmatrix_dmatrix_only_use_S'
    L, D, S = calculate_all_energy(config, seq, chis)
    config['chis'] = None
    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['dmatrix_on'] = True
    config['lmatrix_on'] = True
    config['smatrix_on'] = True
    config['lmatrix_filename'] = 'lmatrix.txt'
    config['dmatrix_filename'] = 'dmatrix.txt'


    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, lmatrix = L,
                dmatrix = D)
    sim.run_eq(1000, 1000, 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def hack_smatrix_as_lmatrix(config, seq, chis):
    root = '/home/erschultz/consistency_check/hack_smatrix_as_lmatrix'
    L, D, S = calculate_all_energy(config, seq, chis)
    config['chis'] = None
    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['diag_chis'] = None
    config['diagonal_on'] = False
    config['dmatrix_on'] = False
    config['lmatrix_on'] = True
    config['smatrix_on'] = False
    config['lmatrix_filename'] = 'lmatrix.txt'


    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, lmatrix = S,
                dmatrix = D)
    sim.run_eq(1000, 1000, 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()


def main():
    dir = '/home/erschultz/consistency_check'
    if not osp.exists(dir):
        os.mkdir(dir, mode=0o755)

    config, seq, chis = setup()
    # baseline(config, seq, chis)
    # baseline_only_d(config, seq, chis)
    # baseline_only_l(config, seq, chis)
    # baseline_energy_on(config, seq, chis)
    smatrix_only(config, seq, chis)
    # lmatrix_dmatrix_only(config, seq, chis)
    # lmatrix_dmatrix_only_use_S(config, seq, chis)
    # hack_smatrix_as_lmatrix(config, seq, chis)
    # smatrix_only_zeros(config, seq, chis)


if __name__ == '__main__':
    main()
