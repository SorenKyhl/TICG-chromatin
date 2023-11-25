import csv
import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import subprocess as sp
import sys
from time import sleep

import numpy as np
import pylib.analysis as analysis
import torch
from data_generation.modify_maxent import get_samples
from max_ent import setup_config
from pylib.Maxent import Maxent
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_D
from pylib.utils.goals import get_goals
from pylib.utils.utils import load_import_log

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.result_summary_plots import \
    predict_chi_in_psi_basis


def fit_max_ent(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar):
    print(sample)
    mode = 'grid'
    dir = f'/home/erschultz/{dataset}/{sub_dir}/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy')).astype(np.float64)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)
    m = len(y)

    root, config = setup_config(dataset, sample, sub_dir, b, phi, v, None, ar)
    k=10
    config['nspecies'] = k
    config['dump_frequency'] = 10000
    config['nbeads'] = m
    config['dump_observables'] = True

    gnn_root = f'{root}-GNN{GNN_ID}-max_ent'
    if osp.exists(gnn_root):
        # shutil.rmtree(gnn_root)
        print('WARNING: root exists')
        return
    os.mkdir(gnn_root, mode=0o755)

    # sleep for random # of seconds so as not to overload gpu
    sleep(np.random.rand()*10)
    model_path = f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{GNN_ID}'
    log_file = osp.join(gnn_root, 'energy.log')
    ofile = osp.join(gnn_root, 'S.npy')
    args_str = f'--m {m} --gnn_model_path {model_path} --sample_path {dir} --bonded_path {root} --sub_dir {sub_dir} --ofile {ofile}'
    # using subprocess gaurantees that pytorch can't keep an GPU vram cached
    sp.run(f"python3 /home/erschultz/TICG-chromatin/scripts/get_params.py {args_str} > {log_file}",
            shell=True)
    S = np.load(ofile)
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


    config['diagonal_on'] = True
    config['dense_diagonal_on'] = True
    config["small_binsize"] = 1
    if len(y) == 512:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 16
        config["big_binsize"] = 28

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
    goals = get_goals(y, None, config)
    params["goals"] = goals
    params['mode'] = 'diag'
    params['iterations'] = 20
    params['parallel'] = 1
    params['equilib_sweeps'] = 10000
    params['production_sweeps'] = 300000
    params['stop_at_convergence'] = True
    params['conv_defn'] = 'normal'
    params['run_longer_at_convergence'] = False

    stdout = sys.stdout
    with open(osp.join(gnn_root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(gnn_root, params, config, seqs, y, fast_analysis=True,
                    final_it_sweeps=300000, mkdir=False, bound_diag_chis=False,
                    initial_chis = all_chis)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout


def fit(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar):
    print(sample)
    mode = 'grid'
    data_dir = f'/home/erschultz/{dataset}'
    if not osp.exists(data_dir):
        data_dir = osp.join('/media/erschultz/1814ae69-5346-45a6-b219-f77f6739171c/', data_dir[1:])
    dir = osp.join(data_dir, f'{sub_dir}/sample{sample}')
    y = np.load(osp.join(dir, 'y.npy')).astype(np.float64)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)
    m = len(y)

    root, config = setup_config(dataset, sample, sub_dir, b, phi, v, None, ar)
    # config_ref = utils.load_json(osp.join(dir, 'config.json'))
    # config['grid_size'] = config_ref['grid_size']
    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['lmatrix_on'] = False
    config['dmatrix_on'] = False
    config['dump_frequency'] = 10000
    config['nbeads'] = m
    config["smatrix_filename"] = "smatrix.txt"

    gnn_root = f'{root}-GNN{GNN_ID}'
    if osp.exists(gnn_root):
        # shutil.rmtree(gnn_root)
        print('WARNING: root exists')
        return
    os.mkdir(gnn_root, mode=0o755)

    # sleep for random # of seconds so as not to overload gpu
    sleep(np.random.rand()*10)
    model_path = f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{GNN_ID}'
    log_file = osp.join(gnn_root, 'energy.log')
    ofile = osp.join(gnn_root, 'S.npy')
    args_str = f'--m {m} --gnn_model_path {model_path} --sample_path {dir} --bonded_path {root} --sub_dir {sub_dir} --ofile {ofile}'
    # using subprocess gaurantees that pytorch can't keep an GPU vram cached
    sp.run(f"python3 /home/erschultz/TICG-chromatin/scripts/max_ent_setup/get_params.py {args_str} > {log_file}",
            shell=True)
    if not osp.exists(ofile):
        print(f'{ofile} does not exist, SKIPPING')
        return
    S = np.load(ofile)

    stdout = sys.stdout
    with open(osp.join(gnn_root, 'log.log'), 'w') as sys.stdout:
        sim = Pysim(gnn_root, config, None, y, randomize_seed = True,
                    mkdir = False, smatrix = S)
        t = sim.run_eq(10000, 300000, 1)
        print(f'Simulation took {np.round(t, 2)} seconds')

        analysis.main_no_maxent(dir=sim.root)
    sys.stdout = stdout

def check(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar):
    mode = 'grid'
    root = f"optimize_{mode}"
    if phi is not None:
        assert v is None
        root = f"{root}_b_{b}_phi_{phi}"
    else:
        root = f"{root}_b_{b}_v_{v}"
    if ar != 1:
        root += f"_spheroid_{ar}"
    data_dir = f'/home/erschultz/{dataset}'
    if not osp.exists(data_dir):
        data_dir = osp.join('/media/erschultz/1814ae69-5346-45a6-b219-f77f6739171c/', data_dir[1:])
    dir = osp.join(data_dir, f'{sub_dir}/sample{sample}')
    root = osp.join(dir, root)
    gnn_root = f'{root}-GNN{GNN_ID}'
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
    mode = 'grid'
    root = f"optimize_{mode}"
    if phi is not None:
        assert v is None
        root = f"{root}_b_{b}_phi_{phi}"
    else:
        root = f"{root}_b_{b}_v_{v}"
    if ar != 1:
        root += f"_spheroid_{ar}"
    dir = f'/home/erschultz/{dataset}/{sub_dir}/sample{sample}'
    root = osp.join(dir, root)
    gnn_root = f'{root}-GNN{GNN_ID}'
    if osp.exists(gnn_root):
        # shutil.rmtree(gnn_root)
        # print(f'removing {gnn_root}')
        if not osp.exists(osp.join(gnn_root, 'equilibration')):
            shutil.rmtree(gnn_root)
            print(f'removing {gnn_root}')
        elif not osp.exists(osp.join(gnn_root, 'production_out')):
            shutil.rmtree(gnn_root)
            print(f'removing {gnn_root}')

def main():
    samples=None
    # dataset='dataset_interp_test'; samples=[1]
    # dataset='dataset_02_04_23';
    # dataset = 'Su2020'; samples=[1013, 1004]
    # dataset = 'dataset_06_29_23'; samples=[81]
    dataset = 'dataset_11_20_23';
    # dataset = 'dataset_06_29_23'; samples = [1,2,3,4,5, 101,102,103,104,105, 601,602,603,604,605]
    mapping = []

    if samples is None:
        samples, _ = get_samples(dataset, train=True, filter_cell_lines='imr90')
        samples = samples[:10]
    print(len(samples))

    GNN_IDs = [614, 615, 616, 617, 618, 619]; b=180; phi=None; v=8; ar=1.5
    for GNN_ID in GNN_IDs:
        for i in samples:
            mapping.append((dataset, i, GNN_ID, f'samples', b, phi, v, ar))

    print(samples)
    print(len(mapping))
    # print(mapping)

    with mp.Pool(15) as p:
        # p.starmap(cleanup, mapping)
        p.starmap(fit, mapping)
#
    for i in mapping:
        #fit_max_ent(*i)
        check(*i)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
