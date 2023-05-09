import json
import multiprocessing as mp
import os
import os.path as osp
import sys

import numpy as np
import optimize_grid
import pylib.analysis as analysis
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from scripts.get_params import GetEnergy


def fit(sample, GNN_ID):
    print(sample)
    mode = 'grid'
    dataset = 'dataset_02_04_23'
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy'))
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)
    nbeads = len(y)

    bonded_config = default.bonded_config
    bonded_config['beadvol'] = 520
    bonded_config['bond_length'] = 16.5
    bonded_config['phi_chromatin'] = 0.06
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

    # get energy
    config['nbeads'] = len(y)
    getenergy = GetEnergy(config = config)
    model_path = f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{GNN_ID}'
    S = getenergy.get_energy_gnn(model_path, dir)
    config["smatrix_filename"] = "smatrix.txt"

    sim = Pysim(f'{root}-GNN{GNN_ID}', config, None, y, randomize_seed = True, overwrite = True,
                smatrix = S)
    sim.run_eq(30000, 500000, 1)

    with utils.cd(sim.root):
        analysis.main_no_maxent()

def main():
    mapping = []
    samples = range(201, 211)
    GNN_IDs = [402]
    for i in samples:
        for GNN_ID in GNN_IDs:
            mapping.append((i, GNN_ID))

    print(mapping)

    with mp.Pool(5) as p:
        p.starmap(fit, mapping)
    # for i in samples:
        # fit(i, GNN_ID)
    # fit(5)



if __name__ == '__main__':
    # soren()
    # modify_soren()
    main()
