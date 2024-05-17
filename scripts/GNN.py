import csv
import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import subprocess as sp
import sys
from time import sleep, time

import numpy as np
import pylib.analysis as analysis
import torch
# from data_generation.modify_maxent import get_samples
# from max_ent import setup_config
from pylib.Maxent import Maxent
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_D
from pylib.utils.goals import get_goals

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.result_summary_plots import \
    predict_chi_in_psi_basis

sys.path.append('/home/erschultz/TICG-chromatin')
from scripts.data_generation.modify_maxent import get_samples
from scripts.max_ent import setup_config, setup_max_ent
from scripts.max_ent_setup.get_params import GetEnergy

ROOT = '/home/erschultz'

def run_GNN(GNN_ID, gnn_root, m, dir, root, sub_dir, use_GPU=True, verbose=True):
    # sleep for random # of seconds so as not to overload gpu
    model_path = osp.join(ROOT, 'sequences_to_contact_maps/results/ContactGNNEnergy',
                        str(GNN_ID))
    log_file = osp.join(gnn_root, 'energy.log')
    ofile = osp.join(gnn_root, 'S.npy')
    if use_GPU:
        sleep_time = (np.random.uniform())*15
        sleep(sleep_time)
        t0 = time()
        args_str = f'--m {m} --gnn_model_path {model_path} --sample_path {dir}'
        args_str += f' --bonded_path {root} --sub_dir {sub_dir} --ofile {ofile}'
        args_str += f' --use_gpu true --verbose {verbose}'
        # using subprocess gaurantees that pytorch can't keep any GPU vram cached
        file = 'TICG-chromatin/scripts/max_ent_setup/get_params.py'
        sp.run(f"python3 {ROOT}/{file} {args_str} > {log_file}",
                shell=True)
    else:
        t0 = time()
        getenergy = GetEnergy(m)
        stdout = sys.stdout
        with open(log_file, 'w') as sys.stdout:
            S = getenergy.get_energy_gnn(model_path, dir,
                                        bonded_path = root,
                                        sub_dir = sub_dir, use_gpu = False,
                                        verbose = verbose)
            np.save(ofile, S)
        sys.stdout = stdout
    tf = time()
    utils.print_time(t0, tf, 'gnn')

    if not osp.exists(ofile):
        print(f'{ofile} does not exist, SKIPPING')
        return

    S = np.load(ofile)
    return S

def fit_max_ent(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar, k=10):
    print(sample)
    mode = 'grid'
    dir, root, config, y = setup_max_ent(dataset, sample, sub_dir, b, phi, v, None,
                                ar, bond_type='gaussian',
                                k=k, contacts_distance=False,
                                k_angle=0, theta_0=180,
                                verbose=False, return_dir=True)
    m = len(y)

    gnn_root = f'{root}-GNN{GNN_ID}'
    if osp.exists(gnn_root):
        # shutil.rmtree(gnn_root)
        print(f'WARNING: root exists: {gnn_root}')
        return
    os.mkdir(gnn_root, mode=0o755)

    S = run_GNN(GNN_ID, gnn_root, m, dir, root, sub_dir)
    if S is None:
        return

    meanDist_S = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
    D = calculate_D(meanDist_S)
    L = S - D
    meanL = np.mean(L)
    L -= meanL
    D += meanL
    all_diag_chis = D[0]

    seqs = epilib.get_pcs(epilib.get_oe(y), k, normalize = True)
    chi = predict_chi_in_psi_basis(seqs, L, verbose = True)
    chi_flat = chi[np.triu_indices(k)]
    config['chis'] = chi


    diag_chis = np.zeros(config['n_small_bins']+config["n_big_bins"])
    left = 0
    right = left + config["small_binsize"]
    bin = 0
    for i in range(config['n_small_bins']):
        diag_chis[bin] = np.mean(all_diag_chis[left:right])
        left += config["small_binsize"]
        right += config["small_binsize"]
        bin += 1
    right = left + config["big_binsize"]
    for i in range(config['n_big_bins']):
        diag_chis[bin] = np.mean(all_diag_chis[left:right])
        left += config["big_binsize"]
        right += config["big_binsize"]
        bin += 1

    config['diag_chis'] = diag_chis
    all_chis = np.concatenate((chi_flat, diag_chis))

    params = default.params
    goals = get_goals(y, seqs, config)
    params["goals"] = goals
    params['iterations'] = 20
    params['equilib_sweeps'] = 10000
    params['production_sweeps'] = 300000
    params['stop_at_convergence'] = True
    params['conv_defn'] = 'normal'

    stdout = sys.stdout
    with open(osp.join(gnn_root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(gnn_root, params, config, seqs, y, fast_analysis=True,
                    final_it_sweeps=300000, mkdir=False, bound_diag_chis=False,
                    initial_chis = all_chis)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout

def setup_GNN(dataset, sample, sub_dir, b, phi, v, ar, GNN_ID):
    dir, root, config = setup_config(dataset, sample, sub_dir, b, phi, v, None,
                                    ar, verbose = False)

    y_file = osp.join(dir, 'y.npy')
    if not osp.exists(y_file):
        raise Exception(f'files does not exist: {y_file}')
    y = np.load(y_file).astype(np.float64)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)
    m = len(y)

    # config_ref = utils.load_json(osp.join(dir, 'config.json'))
    # config['grid_size'] = 200
    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['lmatrix_on'] = False
    config['dmatrix_on'] = False
    config['dump_frequency'] = 1000
    config['nSweeps'] = 3000
    config['nSweeps_eq'] = 1000
    config['nbeads'] = m
    config["umatrix_filename"] = "umatrix.txt"

    gnn_root = f'{root}-GNN{GNN_ID}_test'

    return dir, root, gnn_root, config, y

def fit(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar):
    print(sample)
    mode = 'grid'

    dir, root, gnn_root, config, y = setup_GNN(dataset, sample, sub_dir, b, phi, v, ar, GNN_ID)
    m = len(y)

    if osp.exists(gnn_root):
        return
    os.mkdir(gnn_root, mode=0o755)

    S = run_GNN(GNN_ID, gnn_root, m, dir, root, sub_dir, use_GPU=False)
    if S is None:
        return

    stdout = sys.stdout
    with open(osp.join(gnn_root, 'log.log'), 'w') as sys.stdout:
        sim = Pysim(gnn_root, config, None, y, randomize_seed = True,
                    mkdir = False, umatrix = U)
        t = sim.run_eq(config['nSweeps_eq'], config['nSweeps'], 1)
        print(f'Simulation took {np.round(t, 2)} seconds')

        analysis.main_no_maxent(dir=sim.root)
    sys.stdout = stdout

def check(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar):
    dir, _, gnn_root, config, y = setup_GNN(dataset, sample, sub_dir, b, phi, v, ar, GNN_ID)
    if osp.exists(gnn_root):
        production = osp.join(gnn_root, 'production_out')
        equilibration = osp.join(gnn_root, 'equilibration')
        if osp.exists(production):
            config = utils.load_json(osp.join(production, 'config.json'))
            if not osp.exists(osp.join(gnn_root, 'y.npy')):
                with open(osp.join(production, 'energy.traj'), 'r') as f:
                    last = f.readlines()[-1]
                    it = int(last.split('\t')[0])
                    prcnt = np.round(it / config['nSweeps'] * 100, 1)
                print(f"{gnn_root} in progress: {prcnt}%")
            else:
                print(f"{gnn_root} complete")
        elif not osp.exists(equilibration):
            print(f"{gnn_root} BROKEN")
        else:
            print(f"{gnn_root} CONFUSED")
    else:
        print(f"{gnn_root} not started")

def cleanup(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar):
    dir, _, gnn_root, config, y = setup_GNN(dataset, sample, sub_dir, b, phi, v, ar, GNN_ID)
    if osp.exists(gnn_root):
        remove = False
        if not osp.exists(osp.join(gnn_root, 'equilibration')):
            remove = True
        # elif not osp.exists(osp.join(gnn_root, 'production_out')):
            # remove = True
        elif not osp.exists(osp.join(gnn_root, 'y.npy')):
             remove = True
        if remove:
            shutil.rmtree(gnn_root)
            print(f'removing {gnn_root}')

def rename(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar):
    dir, _, gnn_root, config, y = setup_GNN(dataset, sample, sub_dir, b, phi, v, ar, GNN_ID)
    if osp.exists(gnn_root):
        os.rename(gnn_root, f'{root}_GNN{GNN_ID}')

def main():
    samples=None
    # dataset = 'dataset_HIRES'; samples = [1, 2, 3, 4]
    dataset='dataset_12_06_23';
    # dataset='dataset_HCT116_RAD21_KO'
    # samples = [441, 81, 8, 578, 368, 297, 153, 510, 225]
    # dataset='dataset_12_12_23_imr90'
    # dataset='dataset_02_14_24_imr90'
    # samples = [42, 114, 475, 331, 402, 543]
    # dataset = 'Su2020'; samples=['1013_rescale1', '1004_rescale1']
    # dataset = 'dataset_11_20_23';
    mapping = []

    if samples is None:
        samples = []
        for cell_line in ['imr90']:
            # samples_cell_line, _ = get_samples(dataset, train=True,
            #                                     filter_cell_lines=cell_line)
            # samples.extend(samples_cell_line[:10])
            samples_cell_line, _ = get_samples(dataset, test=True,
                                                filter_cell_lines=cell_line)
            samples.extend(samples_cell_line[:1])

            print(len(samples))

    GNN_IDs = [689]; b=200; phi=None; v=8; ar=1.5
    for GNN_ID in GNN_IDs:
        for i in samples:
            mapping.append((dataset, i, GNN_ID, f'samples', b, phi, v, ar))

    print(f'samples: {samples}')
    print(f'len of mapping: {len(mapping)}')
    # print(mapping)

    # with mp.Pool(1) as p:
        # p.starmap(cleanup, mapping)
        # p.starmap(fit, mapping)

    for i in mapping:
        # fit_max_ent(*i)
        fit(*i)
        # check(*i)
        # rename(*i)
        # cleanup(*i)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
