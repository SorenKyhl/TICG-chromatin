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
from data_generation.modify_maxent import get_samples
from plotting.contact_map import getArgs, plot_all
from pylib.Maxent import Maxent
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import *
from pylib.utils.utils import load_import_log


def run(dir, config, x=None, S=None):
    print(dir)
    if S is not None:
        print(S.shape)
    sim = Pysim(dir, config, x, randomize_seed = False, overwrite = True,
                smatrix = S)
    sim.run_eq(10000, config['nSweeps'], 1)
    analysis.main_no_maxent(dir=dir)


def modify_maxent():
    dataset = 'dataset_06_29_23'
    mapping = []
    dir = f'/home/erschultz/{dataset}/samples/sample1'
    max_ent_root = 'optimize_grid_b_180_v_8_spheroid_1.5-max_ent10'

    for sample in range(1,2):
        dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        dir = osp.join(dir, max_ent_root)
        config = utils.load_json(osp.join(dir, 'iteration30/production_out/config.json'))
        S = np.load(osp.join(dir, 'iteration30/S.npy'))
        x = np.load(osp.join(dir, 'resources/x.npy'))
        chis = np.array(config['chis'])
        diag_chis = calculate_diag_chi_step(config)
        D = calculate_D(diag_chis)
        L = calculate_L(x, chis)
        S2 = calculate_S(L, D)

        assert np.allclose(S, S2, 1e-3, 1e-3), f'{S-S2}'

        target = np.loadtxt(osp.join(dir, 'fitting2/poly8_log_start_5_meanDist_S_fit.txt'))
        delta = target - DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
        D_target = calculate_D(diag_chis + delta)
        L = calculate_L(x, chis)
        S_target = calculate_S(L, D_target)
        check = DiagonalPreprocessing.genomic_distance_statistics(S_target, 'freq')
        assert np.allclose(target, check), f'{target-check}'

        # for arr, label in zip([L,D,S],['L', 'D', 'S']):
        #     meanDist = DiagonalPreprocessing.genomic_distance_statistics(arr, 'freq')
        #     plt.plot(meanDist, label = label)
        # plt.xscale('log')
        # plt.ylabel('mean along diagonal')
        # plt.xlabel('diagonal')
        # plt.legend()
        # plt.savefig(osp.join(dir, 'MeanDist_LDS.png'))
        # plt.close()

        config['nSweeps'] = 300000
        config['dump_observables'] = False
        s_config = config.copy()
        s_config['chis'] = None
        s_config['nspecies'] = 0
        s_config['diag_chis'] = None
        s_config['load_bead_types'] = False
        s_config['bead_type_files'] = None
        s_config['smatrix_filename'] = 'smatrix.txt'
        s_config['diagonal_on'] = False
        s_config['plaid_on'] = True
        s_config['lmatrix_on'] = False
        s_config['dmatrix_on'] = False

        root = osp.join(dir, 'copy')
        mapping.append([root, config.copy(), x, None])

        # root = osp.join(dir, 'copy_S')
        # mapping.append([root, s_config.copy(), None, S])
        #
        # root = osp.join(dir, 'S_target')
        # mapping.append([root, s_config.copy(), None, S_target])

    print(len(mapping))
    if len(mapping) > 0:
        with mp.Pool(min(len(mapping), 5)) as p:
            p.starmap(run, mapping)

def check(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
        aspect_ratio=1, bond_type='gaussian', k=10, contacts_distance=False,
        k_angle=0, theta_0=190):
    root, _, _ = setup_max_ent(dataset, sample, samples, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0, False)

    if osp.exists(root):
        if not osp.exists(osp.join(root, 'iteration30')):
            it=0
            for i in range(30):
                if osp.exists(osp.join(root, f'iteration{i}')):
                    it = i
            prcnt = np.round(it/30*100, 1)
            print(f'{root}: {prcnt}')
        else:
            print(f'{root}: complete')
    else:
        print(f'{root}: not started')


def post_analysis(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
        aspect_ratio=1, bond_type='gaussian', k=10, contacts_distance=False,
        k_angle=0, theta_0=190):
    root, _, _ = setup_max_ent(dataset, sample, samples, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0)
    stdout = sys.stdout
    with open(osp.join(root, 'analysis_log.log'), 'w') as sys.stdout:
        dir = osp.join(root, 'iteration0')
        analysis.main(False, dir, 'all')
    sys.stdout = stdout


def setup_config(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
                aspect_ratio=1.0, bond_type='gaussian', k=None, contacts_distance=False,
                k_angle=0, theta_0=180,
                verbose=True):
    if verbose:
        print(sample)
    data_dir = f'/home/erschultz/{dataset}'
    if not osp.exists(data_dir):
        data_dir = osp.join('/media/erschultz/1814ae69-5346-45a6-b219-f77f6739171c/', data_dir[1:])
    dir = osp.join(data_dir, f'{samples}/sample{sample}')

    bonded_config = default.bonded_config.copy()
    bonded_config['bond_length'] = bl
    assert phi is None or v is None
    if phi is not None:
        bonded_config['phi_chromatin'] = phi
    if v is not None:
        bonded_config['target_volume'] = v
    bonded_config['bond_type'] = bond_type
    bonded_config['update_contacts_distance'] = contacts_distance
    if k_angle != 0:
        bonded_config['angles_on'] = True
        bonded_config['k_angle'] = k_angle
        bonded_config['theta_0'] = theta_0
    if bonded_config['update_contacts_distance']:
        mode = 'distance'
    else:
        mode = 'grid'
    if vb is not None:
        bonded_config['beadvol'] = vb
    else:
        if bonded_config['bond_length'] == 16.5:
            bonded_config['beadvol'] = 520
        if bonded_config['bond_length'] == 117:
            bonded_config['beadvol'] = 26000
        elif bonded_config['bond_length'] == 224:
            bonded_config['beadvol'] = 260000
        else:
            bonded_config['beadvol'] = 130000
    if aspect_ratio != 1.0:
        bonded_config['boundary_type'] = 'spheroid'
        bonded_config['aspect_ratio'] = aspect_ratio

    root = f"optimize_{mode}"
    if phi is not None:
        assert v is None
        root = f"{root}_b_{bl}_phi_{phi}"
    else:
        root = f"{root}_b_{bl}_v_{v}"
    if bonded_config['angles_on']:
        root += f"_angle_{bonded_config['k_angle']}_theta0_{bonded_config['theta_0']}"
    if bonded_config['boundary_type'] == 'spheroid':
        root += f'_spheroid_{aspect_ratio}'
    if bonded_config['bond_type'] == 'DSS':
        root += '_DSS'

    if verbose:
        print(root)
    root = osp.join(dir, root)
    optimum_file = osp.join(root, f'{mode}.txt')
    if osp.exists(optimum_file):
        if mode == 'grid':
            bonded_config['grid_size'] = np.loadtxt(optimum_file)
        elif mode == 'distance':
            bonded_config["distance_cutoff"] = np.loadtxt(optimum_file)
            bonded_config['grid_size'] = 200 # TODO
        angle_file = osp.join(root, 'angle.txt')
        if osp.exists(angle_file):
            bonded_config['k_angle'] = np.loadtxt(angle_file)
            bonded_config['angles_on'] = True
    else:
        root, bonded_config = optimize_grid.main(root, bonded_config, mode)

    config = default.config
    for key in ['beadvol', 'bond_length', 'phi_chromatin', 'target_volume',
                'grid_size', 'distance_cutoff', 'k_angle', 'angles_on', 'theta_0', 'boundary_type',
                'update_contacts_distance', 'aspect_ratio', 'bond_type']:
        if key in bonded_config:
            config[key] = bonded_config[key]

    return dir, root, config

def setup_max_ent(dataset, sample, samples, bl, phi, v, vb,
                aspect_ratio, bond_type, k, contacts_distance,
                k_angle, theta_0, verbose=True):
    if verbose:
        print(sample)
    dir, root, config = setup_config(dataset, sample, samples, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0, verbose)

    y = np.load(osp.join(dir, 'y.npy')).astype(float)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)

    config['nspecies'] = k
    if k > 0:
        config['chis'] = np.zeros((k,k))
    config['dump_frequency'] = 10000
    config['dump_stats_frequency'] = 100
    config['dump_observables'] = True

    # set up diag chis
    config['diagonal_on'] = True
    config['dense_diagonal_on'] = True
    config["small_binsize"] = 1
    if len(y) == 512:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 16
        config["big_binsize"] = 28
    elif len(y) == 256:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 12
        config["big_binsize"] = 16
    elif len(y) == 1024:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 32
        config["big_binsize"] = 30
    elif len(y) == 2560:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 48
        config["big_binsize"] = 52
    elif len(y) == 3270:
        config['n_small_bins'] = 70
        config["n_big_bins"] = 32
        config["big_binsize"] = 100
    else:
        raise Exception(f'Need to specify bin sizes for size={len(y)}')

    config['diag_chis'] = np.zeros(config['n_small_bins']+config["n_big_bins"])

    # config['diag_start'] = 10
    root = osp.join(dir, f'{root}-max_ent{k}_split_binarize')
    if osp.exists(root):
        # shutil.rmtree(root)
        if verbose:
            print('WARNING: root exists')
        return root, None, None
    return root, config, y


def fit(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
        aspect_ratio=1, bond_type='gaussian', k=10, contacts_distance=False,
        k_angle=0, theta_0=180):
    dir = f'/home/erschultz/{dataset}/{samples}/sample{sample}'
    root, config, y = setup_max_ent(dataset, sample, samples, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0)
    if osp.exists(root):
        return
    os.mkdir(root, mode=0o755)

    # get sequences
    seqs = epilib.get_pcs(epilib.get_oe(y), k, binarize=True, split=True)
    # seqs = epilib.get_sequences(y, int(k/2), randomized=True, split=True)

    params = default.params
    goals = epilib.get_goals(y, seqs, config)
    params["goals"] = goals
    params['iterations'] = 30
    params['equilib_sweeps'] = 10000
    params['production_sweeps'] = 300000
    params['stop_at_convergence'] = True
    params['conv_defn'] = 'normal'

    stdout = sys.stdout
    with open(osp.join(root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(root, params, config, seqs, y, fast_analysis=True,
                    final_it_sweeps=300000, mkdir=False, bound_diag_chis=False)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout

def cleanup(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
        aspect_ratio=1, bond_type='gaussian', k=10, contacts_distance=False,
        k_angle=0, theta_0=180):
    dir = f'/home/erschultz/{dataset}/{samples}/sample{sample}'
    root, _, _ = setup_max_ent(dataset, sample, samples, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0, False)

    remove = False
    if osp.exists(root):
        # if not osp.exists(osp.join(root, 'iteration1')):
        #     remove = True
        if not osp.exists(osp.join(root, 'iteration30/tri.png')):
            remove = True
        # remove = True
        if remove:
            print(f'removing {root}')
            shutil.rmtree(root)

def main():
    samples = None
    # dataset = 'dataset_02_04_23'
    # dataset = 'Su2020'; samples = [1013, 1004]
    dataset = 'dataset_11_20_23'
    # samples = [1,2,3,4,5, 101,102,103,104,105, 601,602,603,604,605]
    # dataset='dataset_HCT116_RAD21_KO'; samples=range(1,9)
    # dataset = 'dataset_08_25_23'; samples=[981]
    # samples = sorted(np.random.choice(samples, 12, replace = False))
    # dataset = 'timing_analysis/512'; samples = list(range(1, 16))

    if samples is None:
        samples = []
        for cell_line in ['imr90']:
            samples_cell_line, _ = get_samples(dataset, test=True, filter_cell_lines=cell_line)
            samples.extend(samples_cell_line[:2])
        print(samples)

    mapping = []
    k_angle=0;theta_0=180;b=180;ar=1.5;phi=None;v=8
    contacts_distance=False
    for i in samples:
        for k in [2, 4, 6, 8]:
            mapping.append((dataset, i, f'samples', b, phi, v, None, ar,
                        'gaussian', k, contacts_distance, k_angle, theta_0))
    print('len =', len(mapping))

    with mp.Pool(8) as p:
        # p.starmap(setup_config, mapping)
        p.starmap(fit, mapping)
        # p.starmap(check, mapping)
        # p.starmap(post_analysis, mapping)
        # p.starmap(cleanup, mapping)

    # for i in mapping:
        #fit_max_ent(*i)
        # fit(*i)

if __name__ == '__main__':
    # modify_maxent()
    main()
