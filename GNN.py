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
    y = np.load(osp.join(dir, 'y.npy'))
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
        S = getenergy.get_energy_gnn(model_path, dir, grid_path=osp.join(root, 'grid_size.txt'),
                                    sub_dir = sub_dir)
        config["smatrix_filename"] = "smatrix.txt"

    with open(osp.join(gnn_root, 'log.log'), 'w') as sys.stdout:
        sim = Pysim(gnn_root, config, None, y, randomize_seed = True, mkdir = False,
                    smatrix = S)
        t = sim.run_eq(30000, 500000, 1)
        print(f'Simulation took {np.round(t, 2)} seconds')

        analysis.main_no_maxent(sim.root)
    sys.stdout = stdout

def main():
    dataset='downsampling_analysis'
    mapping = []
    # samples = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100]
    # samples = [1222, 1250, 1279]
    samples = range(201, 211)
    # # samples = [1014]
    # GNN_IDs = [408]
    # for i in samples:
    #     for GNN_ID in GNN_IDs:
    #         mapping.append((dataset, i, GNN_ID))

    mapping = []
    for j in [1]:
        for i in samples:
            mapping.append((dataset, i, 403, f'samples_sim{j}'))
    print(len(mapping))
    print(mapping)

    # with mp.Pool(15) as p:
        # p.starmap(fit, mapping)


    # #
    # print(mapping)
    # print(len(mapping))
    # #
    # with mp.Pool(10) as p:
    #     p.starmap(fit, mapping)
    # for i in samples:
        # fit(dataset, i, GNN_ID)
    fit(dataset, 202, 403, 'samples_sim1')



if __name__ == '__main__':
    # soren()
    # modify_soren()
    main()
