import os
import os.path as osp
import shutil
import sys

import numpy as np
from scipy import optimize

import pylib.utils.epilib as epilib
from pylib.optimize import optimize_config
from pylib.Pysim import Pysim
from pylib.utils import default, utils
from pylib.utils.hic_utils import DiagonalPreprocessing
from scripts.contact_map import plot_max_ent

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder


def main(root, config, mode='grid_angle10'):

    gthic = np.load(osp.join(osp.split(root)[0], 'y.npy'))
    config['nbeads'] = len(gthic)

    if mode.startswith('grid'):
        optimum = optimize_config(config, gthic, 'grid', 0.5, 2.5, root)
        plot_max_ent(root)
        print(f"optimal grid size is: {optimum}")
        with open(osp.join(root, 'grid_size.txt'), 'w') as f:
            f.write(str(optimum))
    elif mode.startswith('angle'):
        assert config['angles_on']
        optimum = optimize_config(config, gthic, 'angle', 0.0, 2.0, root, 'neighbor_10')
        plot_max_ent(root)
        print(f"optimal angle is: {optimum}")
        with open(osp.join(droot, 'angle.txt'), 'w') as f:
            f.write(str(optimum))


    if mode.startswith('grid_angle'):
        os.mkdir('temp', mode=0o755)
        # move grid temporarily
        for file in os.listdir(root):
            shutil.move(osp.join(root, file), osp.join('temp', file))

        config['grid_size'] = optimum
        config['k_angle'] = 0.0
        config['angles_on'] = True

        s = mode[10:]
        optimum = optimize_config(config, gthic, 'angle', 0.0, 2.0, root, f'neighbor_{s}')
        plot_max_ent(root)
        print(f"optimal angle is: {optimum}")
        with open(osp.join(root, 'angle.txt'), 'w') as f:
            f.write(str(optimum))

        # move grid back
        shutil.move('temp', osp.join(root, 'grid'))
        shutil.move(osp.join(root, 'grid/grid_size.txt'), osp.join(root, 'grid_size.txt'))

    return root, config

def check_all_converged():
    dataset = 'dataset_02_04_23'
    for sample in range(201, 283):
        dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        dir = osp.join(dir, 'optimize_grid_b_140_phi_0.06')
        g = np.loadtxt(osp.join(dir, 'grid_size.txt'))
        print(g)

def create_config():
    config = default.bonded_config

    config['beadvol'] = 260000
    config['bond_length'] = 260
    config['phi_chromatin'] = 0.06
    config['grid_size'] = 222.5
    # config['bond_type'] = 'DSS'
    config['k_angle'] = 0.0
    config['angles_on'] = False

    return config

if __name__ == "__main__":
    config = create_config()
    mode = 'grid_angle20'
    root = f"optimize_{mode}"
    root = f"{root}_b_{config['bond_length']}_phi_{config['phi_chromatin']}"
    root = osp.join('/home/erschultz/dataset_test/samples/sample5002', root)
    main(root, config, mode)
    # check_all_converged()
