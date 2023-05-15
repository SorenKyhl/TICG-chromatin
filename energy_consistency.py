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


def setup(seed):
    config = default.config
    m=512
    config['nbeads'] = m
    config['nspecies'] = 4
    config['seed'] = seed
    config['bead_type_files'] = [f'pcf{i}.txt' for i in range(1, config['nspecies']+1)]
    config['track_contactmap'] = False
    config['beadvol'] = 130000
    config['bond_length'] = 140
    config['phi_chromatin'] = 0.03
    config['grid_size'] = 210
    config['dump_frequency'] = 1
    config['dump_stats_frequency'] = 1
    config['equilibSweeps'] = 1000
    config['nSweeps'] = 1000

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
    rng = np.random.default_rng(seed)
    seq = rng.choice((0,1), size=(m,4))
    print('psi')
    print(seq[0])

    chis = rng.normal(size=(4,4))*2
    print('chis')
    print(chis)
    config['chis'] = chis.tolist()

    return config, seq, chis

def baseline(dir, config, seq, chis):
    root = f'{dir}/baseline'
    config['smatrix_on'] = False
    config['dmatrix_on'] = False
    config['lmatrix_on'] = False

    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_only_l(dir, config, seq, chis):
    root = f'{dir}/baseline_only_l'
    config['smatrix_on'] = False
    config['dmatrix_on'] = False


    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_only_d(dir, config, seq, chis):
    root = f'{dir}/baseline_only_d'
    config['smatrix_on'] = False
    config['lmatrix_on'] = False

    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_energy_on(dir, config, seq, chis):
    root = f'{dir}/baseline_energy_on'
    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_energy_on_no_gridmove(dir, config, seq, chis):
    root = f'{dir}/baseline_energy_on_no_gridmove'

    config['gridmove_on'] = False

    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_energy_on_only_trans(dir, config, seq, chis):
    root = f'{dir}/baseline_energy_on_only_trans'

    config['pivot_on'] = False
    config['crankshaft_on'] = False
    config['gridmove_on'] = False


    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def baseline_energy_on_only_pivot(dir, config, seq, chis):
    root = f'{dir}/baseline_energy_on_only_pivot'

    config['translation_on'] = False
    config['crankshaft_on'] = False
    config['gridmove_on'] = False


    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()


def baseline_energy_on_only_trans_grid(dir, config, seq, chis):
    root = f'{dir}/baseline_energy_on_only_trans_grid'

    config['pivot_on'] = False
    config['crankshaft_on'] = False


    sim = Pysim(root, config, seq, None, randomize_seed = False, overwrite = True)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
        analysis.main_no_compare()

def smatrix_only(dir, config, seq, chis):
    root = f'{dir}/smatrix_only'
    print(root)
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
    #
    # S_prime = convert_L_to_Lp(S)
    #
    # S_prime2 = np.loadtxt(osp.join(root, 'equilibration/smatrix_prime.txt'))
    #
    # print(np.max(S_prime - S_prime2))

    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, smatrix = S)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def smatrix_only_trans(dir, config, seq, chis):
    root = f'{dir}/smatrix_only_trans'
    L, D, S = calculate_all_energy(config, seq, chis)

    config = smatrix_mode(config)
    config['pivot_on'] = False
    config['crankshaft_on'] = False
    config['gridmove_on'] = False


    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, smatrix = S)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def smatrix_only_trans_grid(dir, config, seq, chis):
    root = f'{dir}/smatrix_only_trans_grid'
    L, D, S = calculate_all_energy(config, seq, chis)

    config = smatrix_mode(config)
    config['pivot_on'] = False
    config['crankshaft_on'] = False

    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, smatrix = S)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def smatrix_only_no_gridmove(dir, config, seq, chis):
    root = f'{dir}/smatrix_only_no_gridmove'
    L, D, S = calculate_all_energy(config, seq, chis)

    config = smatrix_mode(config)
    config['gridmove_on'] = False


    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, smatrix = S)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def smatrix_only_pivot(dir, config, seq, chis):
    root = f'{dir}/smatrix_only_pivot'
    L, D, S = calculate_all_energy(config, seq, chis)

    config = smatrix_mode(config)
    config['translation_on'] = False
    config['crankshaft_on'] = False
    config['gridmove_on'] = False


    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, smatrix = S)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()




def smatrix_only_zeros(dir, config, seq, chis):
    root = f'{dir}/smatrix_only_zeros'
    S = np.zeros((512, 512))
    # print(S)
    config = smatrix_mode(config)
    sim = Pysim(root, config, None, None, randomize_seed = False, overwrite = True, smatrix = S)
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def lmatrix_dmatrix_only(dir, config, seq, chis):
    root = f'{dir}/lmatrix_dmatrix_only'
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
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def lmatrix_dmatrix_only_use_S(dir, config, seq, chis):
    root = f'{dir}/lmatrix_dmatrix_only_use_S'
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
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()

def hack_smatrix_as_lmatrix(dir, config, seq, chis):
    root = f'{dir}/hack_smatrix_as_lmatrix'
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
    sim.run_eq(config['equilibSweeps'], config['nSweeps'], 1)

    with utils.cd(sim.root):
      analysis.main_no_compare()


def main():
    seed=13
    dir = f'/home/erschultz/consistency_check_seed{seed}'
    if not osp.exists(dir):
        os.mkdir(dir, mode=0o755)

    config, seq, chis = setup(seed)
    # baseline(dir, config, seq, chis)
    # baseline_only_d(dir, config, seq, chis)
    # baseline_only_l(dir, config, seq, chis)
    # baseline_energy_on(dir, config, seq, chis)
    # baseline_energy_on_only_trans(dir, config, seq, chis)
    smatrix_only(dir, config, seq, chis)
    # smatrix_only_trans_grid(dir, config, seq, chis)
    # smatrix_only_trans(dir, config, seq, chis)
    # lmatrix_dmatrix_only(dir, config, seq, chis)
    # lmatrix_dmatrix_only_use_S(dir, config, seq, chis)
    # hack_smatrix_as_lmatrix(dir, config, seq, chis)
    # smatrix_only_zeros(dir, config, seq, chis)
    # smatrix_only_no_gridmove(dir, config, seq, chis)
    # baseline_energy_on_no_gridmove(dir, config, seq, chis)
    # baseline_energy_on_only_trans_grid(dir, config, seq, chis)
    # baseline_energy_on_only_pivot(dir, config, seq, chis)
    # smatrix_only_pivot(dir, config, seq, chis)

if __name__ == '__main__':
    main()
