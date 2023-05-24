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

def fit(sample):
    print(sample)
    mode = 'grid'
    dataset = 'Su2020'
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy'))
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)

    bonded_config = default.bonded_config
    bonded_config['bond_length'] = 140
    bonded_config['phi_chromatin'] = 0.03
    if bonded_config['bond_length'] == 16.5:
        bonded_config['beadvol'] = 520
    else:
        bonded_config['beadvol'] = 130000
    root = f"optimize_{mode}"
    root = f"{root}_b_{bonded_config['bond_length']}_phi_{bonded_config['phi_chromatin']}"
    print(root)
    root = osp.join(dir, root)
    if osp.exists(root):
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
    k = 10
    config['nspecies'] = k
    config['chis'] = np.zeros((k,k))
    config['dump_frequency'] = 10000

    # set up diag chis
    config['diagonal_on'] = True
    config['dense_diagonal_on'] = True
    config['n_small_bins'] = 64
    config["small_binsize"] = 1
    if len(y) == 1024:
        config["n_big_bins"] = 32
        config["big_binsize"] = 30
    elif len(y) == 512:
        config["n_big_bins"] = 16
        config["big_binsize"] = 28
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
    params['iterations'] = 15
    params['parallel'] = 1
    params['equilib_sweeps'] = 30000
    params['production_sweeps'] = 300000


    stdout = sys.stdout
    with open(osp.join(root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(root, params, config, seqs, y,
                    final_it_sweeps=500000, mkdir=False)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout

def main():
    # with mp.Pool(8) as p:
        # p.map(fit, range(1001, 1211))
    # for i in range(1001, 1211):
        # fit(i)
    fit(1013)



if __name__ == '__main__':
    # soren()
    # modify_soren()
    main()
