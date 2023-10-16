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
import optimize_grid
import pylib.analysis as analysis
import torch
from max_ent import setup_config
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils

from scripts.data_generation.modify_maxent import get_samples
from scripts.get_params import GetEnergy

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_import_log


def fit(dataset, sample, GNN_ID, sub_dir, b, phi, v, ar):
    print(sample)
    mode = 'grid'
    dir = f'/home/erschultz/{dataset}/{sub_dir}/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy')).astype(np.float64)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)
    m = len(y)

    root, config = setup_config(dataset, sample, sub_dir, b, phi, v, None, ar)
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
    sp.run(f"python3 /home/erschultz/TICG-chromatin/scripts/get_params.py {args_str} > {log_file}",
            shell=True)
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
    dir = f'/home/erschultz/{dataset}/{sub_dir}/sample{sample}'
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


def main():
    samples=None
    # dataset='dataset_02_04_23';
    # dataset = 'Su2020'; samples=[1013, 1004]
    # dataset = 'dataset_06_29_23'; samples = [2, 103, 604]
    dataset = 'dataset_09_28_23'
    # dataset='dataset_09_28_23_s_100_cutoff_0.01';
    # dataset='dataset_09_28_23_s_10_cutoff_0.08';
    # dataset='dataset_09_28_23_s_1_cutoff_0.36';
    # dataset = 'dataset_06_29_23'; samples = [1,2,3,4,5, 101,102,103,104,105, 601,602,603,604,605]
    mapping = []

    if samples is None:
        samples, _ = get_samples(dataset, train=True)
        samples = samples[:5]
    print(len(samples))

    GNN_IDs = [539]; b=180; phi=None; v=8; ar=1.5
    for GNN_ID in GNN_IDs:
        for i in samples:
            mapping.append((dataset, i, GNN_ID, f'samples', b, phi, v, ar))

    print(samples)
    print(len(mapping))
    # print(mapping)

    # with mp.Pool(2) as p:
        # p.starmap(cleanup, mapping)
        # p.starmap(fit, mapping)

    for i in mapping:
        check(*i)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
