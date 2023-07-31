import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import numpy as np
import optimize_grid
from pylib.Maxent import Maxent
from pylib.utils import default, epilib, utils
from scripts.get_params import GetSeq


def soren():
    dir = '/home/erschultz/dataset_test/samples/sample5003'

    y = np.load(osp.join(dir, 'y.npy'))
    seqs = np.load(osp.join(dir, 'x_soren.npy')).T
    with open(osp.join(dir, 'config.json')) as f:
        config = json.load(f)
    config['dump_frequency'] = 5000
    # config['dump_stats_frequency'] = 10
    config["lmatrix_on"] = True
    config["smatrix_on"] = True
    config["dmatrix_on"] = True
    config['seed'] = 12

    goals = epilib.get_goals(y, seqs, config)
    params = default.params
    params["goals"] = goals
    params['iterations'] = 2
    params['parallel'] = 5
    params['production_sweeps'] = 5000
    # params['equilib_sweeps'] = 50000

    me = Maxent(osp.join(dir, 'soren_me_0-S'), params, config, seqs, y,
                final_it_sweeps=0)
    me.fit()

def modify_soren():
    dataset = 'dataset_test'
    sample = 5003
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy'))

    root = osp.join(dir, 'soren_me_0-seq')
    print(root)
    config = utils.load_json(osp.join(dir, 'soren_me_0/iteration10/config.json'))
    k = config['nspecies']
    config['chis'] = np.zeros((k,k))
    config['diag_chis'] = np.zeros(len(config['diag_chis']))

    # get sequences
    config['nbeads'] = len(y)
    getSeq = GetSeq(config = config)
    seqs = getSeq.get_PCA_seq(epilib.get_oe(y), normalize = True, smooth = False)
    # seqs = epilib.get_sequences(y, config['nspecies'], randomized=True)

    params = default.params
    goals = epilib.get_goals(y, seqs, config)
    params["goals"] = goals
    # params['iterations'] = 1
    params['parallel'] = 5
    params['production_sweeps'] = 70000
    # params['equilib_sweeps'] = 1000

    me = Maxent(root, params, config, seqs, y,
                final_it_sweeps=500000)
    me.fit()

def setup_config(dataset, sample, samples='samples'):
    mode = 'grid'
    print(sample, mode)
    dir = f'/home/erschultz/{dataset}/{samples}/sample{sample}'

    bonded_config = default.bonded_config
    bonded_config['bond_length'] = 140
    bonded_config['phi_chromatin'] = 0.03
    if bonded_config['bond_length'] == 16.5:
        bonded_config['beadvol'] = 520
    if bonded_config['bond_length'] == 117:
        bonded_config['beadvol'] = 26000
    elif bonded_config['bond_length'] == 224:
        bonded_config['beadvol'] = 260000
    else:
        bonded_config['beadvol'] = 130000
    root = f"optimize_{mode}"
    root = f"{root}_b_{bonded_config['bond_length']}_phi_{bonded_config['phi_chromatin']}"
    print(root)
    root = osp.join(dir, root)
    if osp.exists(osp.join(root, 'grid_size.txt')):
        bonded_config['grid_size'] = np.loadtxt(osp.join(root, 'grid_size.txt'))
        angle_file = osp.join(root, 'angle.txt')
        if osp.exists(angle_file):
            bonded_config['k_angle'] = np.loadtxt(angle_file)
            bonded_config['angles_on'] = True
    else:
        root, bonded_config = optimize_grid.main(root, bonded_config, mode)

    config = default.config
    for key in ['beadvol', 'bond_length', 'phi_chromatin', 'grid_size',
                'k_angle', 'angles_on']:
        config[key] = bonded_config[key]

    return root, config


def fit(dataset, sample, samples='samples'):
    print(sample)
    mode = 'grid'
    dir = f'/home/erschultz/{dataset}/{samples}/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy'))
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)

    root, config = setup_config(dataset, sample, samples)

    k = 10
    config['nspecies'] = k
    config['chis'] = np.zeros((k,k))
    config['dump_frequency'] = 10000
    config['dump_observables'] = True

    # set up diag chis
    config['diagonal_on'] = True
    config['dense_diagonal_on'] = True
    config["small_binsize"] = 1
    if len(y) == 1024:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 32
        config["big_binsize"] = 30
    elif len(y) == 512:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 16
        config["big_binsize"] = 28
    elif len(y) == 3270:
        config['n_small_bins'] = 70
        config["n_big_bins"] = 32
        config["big_binsize"] = 100

    config['diag_chis'] = np.zeros(config['n_small_bins']+config["n_big_bins"])

    root = osp.join(dir, f'{root}-max_ent{k}')
    if osp.exists(root):
        # shutil.rmtree(root)
        print('WARNING: root exists')
        return
    os.mkdir(root, mode=0o755)

    # get sequences
    seqs = epilib.get_pcs(epilib.get_oe(y), k, normalize = True)
    # seqs = epilib.get_sequences(y, k, randomized=True)

    params = default.params
    goals = epilib.get_goals(y, seqs, config)
    params["goals"] = goals
    params['iterations'] = 20
    params['parallel'] = 1
    params['equilib_sweeps'] = 30000
    params['production_sweeps'] = 400000

    stdout = sys.stdout
    with open(osp.join(root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(root, params, config, seqs, y,
                    final_it_sweeps=0, mkdir=False)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout

def main():
    # dataset = 'dataset_05_31_23'; samples = list(range(1137, 1214))
    # dataset = 'downsampling_analysis'; samples = list(range(201, 211))
    # dataset = 'dataset_02_04_23'; samples = list(range(201, 270))
    # dataset = 'Su2020'; samples = [1013]
    # dataset = 'dataset_04_05_23'; samples = list(range(1211, 1288))
    # dataset = 'dataset_06_29_23'; samples = list(range(1, 16))
    # samples = sorted(np.random.choice(samples, 12, replace = False))
    dataset = 'timing_analysis/512'; samples = list(range(1, 16))

    mapping = []
    # for j in [10]:
    for i in samples:
        mapping.append((dataset, i, f'samples'))
    print(len(mapping))
    print(mapping)

    with mp.Pool(15) as p:
        p.starmap(fit, mapping)
    # for i in samples:
    #     setup_config(dataset, i, 'samples')

    # dataset = 'Su2020'
    # fit(dataset, 1013, 'samples')



if __name__ == '__main__':
    # soren()
    # modify_soren()
    main()
