import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import optimize_grid
import pylib.analysis as analysis
from max_ent import setup_config
from pylib.Maxent import Maxent
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import *
from scripts.contact_map import getArgs, plot_all
from scripts.data_generation.modify_maxent import get_samples

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_import_log


def fit(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
        aspect_ratio=1, bond_type='gaussian', k=10, contact_distance=False,
        k_angle=0, theta_0=180, gamma=1):
    print(sample)
    dir = f'/home/erschultz/{dataset}/{samples}/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy')).astype(float)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)

    root, config = setup_config(dataset, sample, samples, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contact_distance,
                                k_angle, theta_0)

    config['nspecies'] = k
    if k > 0:
        config['chis'] = np.zeros((k,k))
    config['dump_frequency'] = 10000
    config['dump_stats_frequency'] = 10
    config['dump_observables'] = True

    # set up diag chis
    config['diagonal_on'] = True
    config['dense_diagonal_on'] = True
    config["small_binsize"] = 1
    assert len(y) == 512
    config['n_small_bins'] = 64
    config["n_big_bins"] = 16
    config["big_binsize"] = 28
    config['diag_chis'] = np.zeros(config['n_small_bins']+config["n_big_bins"])

    # config['diag_start'] = 10
    root = osp.join(dir, f'{root}-max_ent{k}-gd_gamma{gamma}')
    if osp.exists(root):
        # shutil.rmtree(root)
        print('WARNING: root exists')
        return
    os.mkdir(root, mode=0o755)

    # get sequences
    seqs = epilib.get_pcs(epilib.get_oe(y), k, normalize = True)

    params = default.params
    goals = epilib.get_goals(y, seqs, config)
    params["goals"] = goals
    params['iterations'] = 20
    params['equilib_sweeps'] = 10000
    params['production_sweeps'] = 350000
    params['stop_at_convergence'] = True
    params['conv_defn'] = 'normal'
    params['method'] = 'g'
    params['gamma'] = gamma

    stdout = sys.stdout
    with open(osp.join(root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(root, params, config, seqs, y, fast_analysis=True,
                    final_it_sweeps=350000, mkdir=False, bound_diag_chis=False)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout


def main():
    samples = None
    dataset = 'dataset_06_29_23';

    if samples is None:
        samples, _ = get_samples(dataset, train=True)
        samples = samples[:1]
        print(samples)

    mapping = []
    k_angle=0;theta_0=180;b=180;ar=1.5;phi=None;v=8;k=10
    for i in samples:
        for gamma in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
            mapping.append((dataset, i, f'samples', b, phi, v, None, ar,
                        'gaussian', k, False, k_angle, theta_0, gamma))
    print('len =', len(mapping))

    with mp.Pool(7) as p:
        # p.starmap(setup_config, mapping)
        p.starmap(fit, mapping)
        # p.starmap(check, mapping)
        # p.starmap(cleanup, mapping)

if __name__ == '__main__':
    # modify_maxent()
    main()
