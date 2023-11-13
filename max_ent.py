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
from pylib.Maxent import Maxent
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import *
from scripts.contact_map import getArgs, plot_all
from scripts.data_generation.modify_maxent import get_samples

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_import_log


def run(dir, config, x=None, S=None):
    print(dir)
    print(S.shape)
    sim = Pysim(dir, config, x, randomize_seed = False, overwrite = True,
                smatrix = S)
    sim.run_eq(10000, config['nSweeps'], 1)
    analysis.main_no_maxent(dir=dir)


def modify_maxent():
    dataset = 'dataset_02_04_23'
    mapping = []
    dir = f'/home/erschultz/{dataset}/samples/sample201'
    dir = osp.join(dir, 'optimize_grid_b_261_phi_0.01-max_ent10')
    config = utils.load_json(osp.join(dir, 'iteration30/production_out/config.json'))
    S_ref = np.load(osp.join(dir, 'iteration30/S.npy'))
    meanDist_S_ref = DiagonalPreprocessing.genomic_distance_statistics(S_ref, 'freq')
    diag_chis_init1 = meanDist_S_ref -  meanDist_S_ref[1]
    diag_chis_init1[0] = 0
    print(diag_chis_init1[:10])
    diag_chis_init1_dense = diag_chi_step_to_dense(diag_chis_init1, 64, 1, 16, 28)
    np.save(osp.join(dir, 'diag_chis_init1.npy'), diag_chis_init1_dense)

    diag_chis_init2 = meanDist_S_ref -  meanDist_S_ref[3]
    diag_chis_init2[0:2] = 0
    print(diag_chis_init2[:10])
    return

    for sample in range(201,205):
        dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        dir = osp.join(dir, 'optimize_grid_b_261_phi_0.01-max_ent10')
        config = utils.load_json(osp.join(dir, 'iteration30/production_out/config.json'))
        S = np.load(osp.join(dir, 'iteration30/S.npy'))
        x = np.load(osp.join(dir, 'resources/x.npy'))
        chis = np.array(config['chis'])
        diag_chis = calculate_diag_chi_step(config)
        D = calculate_D(diag_chis)
        L = calculate_L(x, chis)
        S2 = calculate_S(L, D)

        meanDist_S = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
        meanDist_L = DiagonalPreprocessing.genomic_distance_statistics(L, 'freq')
        S_center = S - np.mean(S)
        S_center2 = S - np.mean(S.diagonal())
        S_center3 = S - diag_chis[1]
        S_meanDist = calculate_D(meanDist_S)

        diag_chis_init1 = meanDist_S - meanDist_S[1]
        diag_chis_init1[0] = 0
        D_init1 = calculate_D(diag_chis_init1)

        diag_chis_init2 = meanDist_S -  meanDist_S[3]
        diag_chis_init2[0:2] = 0
        D_init2 = calculate_D(diag_chis_init2)

        meanDist_delta = meanDist_S_ref - meanDist_S
        D_delta = calculate_D(meanDist_delta)
        S_delta = S + D_delta
        L_meanDist = calculate_D(meanDist_L)

        assert np.allclose(S, S2, 1e-3, 1e-3), f'{S-S2}'

        for arr, label in zip([L,D,S],['L', 'D', 'S']):
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(arr, 'freq')
            plt.plot(meanDist, label = label)
        plt.xscale('log')
        plt.ylabel('mean along diagonal')
        plt.xlabel('diagonal')
        plt.legend()
        plt.savefig(osp.join(dir, 'MeanDist_LDS.png'))
        plt.close()


        config['nSweeps'] = 350000
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

        # root = osp.join(dir, 'copy')
        # mapping.append([root, config.copy(), x, None])

        # root = osp.join(dir, 'copy_S')
        # mapping.append([root, s_config.copy(), None, S])

        # root = osp.join(dir, 'copy_S_center')
        # mapping.append([root, s_config.copy(), None, S_center])
        # #
        # root = osp.join(dir, 'copy_S_center2')
        # mapping.append([root, s_config.copy(), None, S_center2])
        #
        # root = osp.join(dir, 'copy_S_center3')
        # mapping.append([root, s_config.copy(), None, S_center3])

        root = osp.join(dir, 'copy_D_init1')
        mapping.append([root, s_config.copy(), None, D_init1])

        root = osp.join(dir, 'copy_D_init2')
        mapping.append([root, s_config.copy(), None, D_init2])

        # root = osp.join(dir, 'copy_S_meanDist')
        # mapping.append([root, s_config.copy(), None, S_meanDist])

        # root = osp.join(dir, 'copy_S_delta')
        # mapping.append([root, s_config.copy(), None, S_delta])


        # root = osp.join(dir, 'copy_D')
        # mapping.append([root, s_config.copy(), None, D])

    print(len(mapping))
    if len(mapping) > 1:
        with mp.Pool(min(len(mapping), 5)) as p:
            p.starmap(run, mapping)

def setup_config(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
                aspect_ratio=1.0, bond_type='gaussian', k=None, contact_distance=False,
                k_angle=0, theta_0=180,
                verbose=True):
    if verbose:
        print(sample)
    dir = f'/home/erschultz/{dataset}/{samples}/sample{sample}'

    bonded_config = default.bonded_config.copy()
    bonded_config['bond_length'] = bl
    assert phi is None or v is None
    if phi is not None:
        bonded_config['phi_chromatin'] = phi
    if v is not None:
        bonded_config['target_volume'] = v
    bonded_config['bond_type'] = bond_type
    bonded_config['update_contacts_distance'] = contact_distance
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
    if bonded_config['update_contacts_distance']:
        root += '_distance'

    if verbose:
        print(root)
    root = osp.join(dir, root)
    optimimum_file = osp.join(root, f'{mode}.txt')
    if osp.exists(optimimum_file):
        if mode == 'grid':
            bonded_config['grid_size'] = np.loadtxt(optimimum_file)
        elif mode == 'distance':
            bonded_config["distance_cutoff"] = np.loadtxt(optimimum_file)
        angle_file = osp.join(root, 'angle.txt')
        if osp.exists(angle_file):
            bonded_config['k_angle'] = np.loadtxt(angle_file)
            bonded_config['angles_on'] = True
    else:
        root, bonded_config = optimize_grid.main(root, bonded_config, mode)

    config = default.config
    for key in ['beadvol', 'bond_length', 'phi_chromatin', 'target_volume',
                'grid_size', 'k_angle', 'angles_on', 'theta_0', 'boundary_type',
                'update_contacts_distance', 'aspect_ratio', 'bond_type']:
        if key in bonded_config:
            config[key] = bonded_config[key]

    return root, config

def fit(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
        aspect_ratio=1, bond_type='gaussian', k=10, contact_distance=False,
        k_angle=0, theta_0=180):
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
    config['update_contacts_distance'] = True
    config['distance_cutoff'] = config['grid_size']

    # config['diag_start'] = 10
    root = osp.join(dir, f'{root}-max_ent{k}-distance')
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
    params['iterations'] = 2
    params['parallel'] = 1
    params['equilib_sweeps'] = 1000
    params['production_sweeps'] = 3000
    params['stop_at_convergence'] = True
    params['conv_defn'] = 'normal'
    params['run_longer_at_convergence'] = False

    stdout = sys.stdout
    with open(osp.join(root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(root, params, config, seqs, y, fast_analysis=True,
                    final_it_sweeps=3000, mkdir=False, bound_diag_chis=False)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout

def cleanup(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
        aspect_ratio=1, bond_type='gaussian', k=10, contact_distance=False,
        k_angle=0, theta_0=180):
    print(sample)
    dir = f'/home/erschultz/{dataset}/{samples}/sample{sample}'
    root, _ = setup_config(dataset, sample, samples, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contact_distance,
                                k_angle, theta_0,
                                verbose=False)

    root = osp.join(dir, f'{root}-max_ent{k}')
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

def check(dataset, sample, samples='samples', bl=140, phi=0.03, v=None, vb=None,
        aspect_ratio=1, bond_type='gaussian', k=10, contact_distance=False,
        k_angle=0, theta_0=190):
    dir = f'/home/erschultz/{dataset}/{samples}/sample{sample}'
    root, _ = setup_config(dataset, sample, samples, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contact_distance,
                                verbose=False)

    root = osp.join(dir, f'{root}-max_ent{k}')
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



def main():
    samples = None
    # dataset = 'dataset_05_31_23'; samples = list(range(1137, 1214))
    # dataset = 'downsampling_analysis'; samples = list(range(201, 211))
    # dataset = 'dataset_02_04_23'
    # dataset = 'Su2020'; samples = [1013, 1004]
    # dataset = 'dataset_04_28_23'; samples = [1,2,3,4,5,324,981,1753,1936,2834,3464]
    # dataset = 'dataset_04_05_23'; samples = list(range(1211, 1288))
    dataset = 'dataset_06_29_23';
    # samples = [1,2,3,4,5, 101,102,103,104,105,
    #                                             601,602,603,604,605]
    # dataset = 'dataset_08_25_23'; samples=[981]
    # dataset='dataset_09_28_23_s_100_cutoff_0.01'; samples = [1191, 1478, 4990, 5612, 3073, 1351, 4128, 2768, 9627, 4127, 1160, 8932, 2929, 7699, 6629]
    # samples = sorted(np.random.choice(samples, 12, replace = False))
    # dataset = 'timing_analysis/512'; samples = list(range(1, 16))

    if samples is None:
        samples, _ = get_samples(dataset, train=True)
        samples = samples[:1]
        print(samples)

    mapping = []
    k_angle=0;theta_0=180;b=180;ar=1.5;phi=None;v=8
    for i in samples:
        for k in [10]:
            mapping.append((dataset, i, f'samples', b, phi, v, None, ar,
                        'gaussian', k, False, k_angle, theta_0))
    print('len =', len(mapping))

    with mp.Pool(1) as p:
        # p.starmap(setup_config, mapping)
        p.starmap(fit, mapping)
        # p.starmap(check, mapping)
        # p.starmap(cleanup, mapping)

if __name__ == '__main__':
    # modify_maxent()
    main()
