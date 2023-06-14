import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import numpy as np
import optimize_grid
import pylib.analysis as analysis
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from scripts.get_params import GetEnergy


def fit(dataset, sample, GNN_ID, sub_dir='samples'):
    print(sample)
    mode = 'grid'
    dir = f'/home/erschultz/{dataset}/{sub_dir}/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy')).astype(np.float64)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)
    nbeads = len(y)

    bonded_config = default.bonded_config
    bonded_config['bond_length'] = 140
    bonded_config['phi_chromatin'] = 0.03
    if bonded_config['bond_length'] == 16.5:
        bonded_config['beadvol'] = 520
    else:
        bonded_config['beadvol'] = 130000
    bonded_config["nSweeps"] = 20000
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
    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['lmatrix_on'] = False
    config['dmatrix_on'] = False
    config['dump_frequency'] = 10000

    gnn_root = f'{root}-GNN{GNN_ID}'
    if osp.exists(gnn_root):
        # shutil.rmtree(gnn_root)
        print('WARNING: root exists')
        return
    os.mkdir(gnn_root, mode=0o755)

    stdout = sys.stdout
    with open(osp.join(gnn_root, 'energy.log'), 'w') as sys.stdout:
        # get energy
        config['nbeads'] = len(y)
        getenergy = GetEnergy(config = config)
        model_path = f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{GNN_ID}'
        S = getenergy.get_energy_gnn(model_path, dir,
                                    bonded_path=root,
                                    sub_dir = sub_dir)
        config["smatrix_filename"] = "smatrix.txt"

    with open(osp.join(gnn_root, 'log.log'), 'w') as sys.stdout:
        sim = Pysim(gnn_root, config, None, y, randomize_seed = True,
                    mkdir = False, smatrix = S)
        t = sim.run_eq(30000, 500000, 1)
        print(f'Simulation took {np.round(t, 2)} seconds')

        analysis.main_no_maxent(sim.root)
    sys.stdout = stdout

def main():
    # dataset='downsampling_analysis'; samples = range(201, 211)
    # dataset='dataset_02_04_23'; samples = range(211, 221)
    # dataset='dataset_04_10_23'; samples = range(1001, 1011)
    dataset='dataset_04_05_23'; samples = range(1001, 1011)
    # dataset='dataset_05_28_23'; samples = [324, 981, 1936, 2834, 3464]
    # dataset = 'Su2020'; samples=[1013]
    mapping = []

    GNN_IDs = [421, 422]
    for GNN_ID in GNN_IDs:
        for i in samples:
            mapping.append((dataset, i, GNN_ID))


    # for j in [1]:
    # for i in samples:
        # mapping.append((dataset, i, 403, f'samples'))
    print(len(mapping))
    print(mapping)

    with mp.Pool(10) as p:
        p.starmap(fit, mapping)

    # fit(*mapping[0])


if __name__ == '__main__':
    # soren()
    # modify_soren()
    main()
