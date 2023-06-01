import os
import os.path as osp
import shutil
import sys

import numpy as np
import pylib.utils.epilib as epilib
from pylib.optimize import optimize_config
from pylib.Pysim import Pysim
from pylib.utils import default, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from scipy import optimize
from scripts.contact_map import plot_max_ent

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder


def main(root, config, mode='grid_angle10'):

    gthic = np.load(osp.join(osp.split(root)[0], 'y.npy')).astype(float)
    config['nbeads'] = len(gthic)

    if mode.startswith('grid'):
        optimum = optimize_config(config, gthic, 'grid', 0.7, 2.0, root)
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

    config['beadvol'] = 26000
    config['bond_length'] = 117
    config['phi_chromatin'] = 0.01
    config['grid_size'] = 150
    # config['bond_type'] = 'DSS'
    config['k_angle'] = 0.0
    config['angles_on'] = False

    return config

if __name__ == "__main__":
    config = create_config()
    dir = '/home/erschultz/Su2020/samples/sample3'
    # config = utils.load_json(osp.join(dir, 'config.json'))
    config['track_contactmap'] = False
    # config['bead_type_files'] = [osp.join(dir, f'seq{i}.txt') for i in range(10)]
    mode = 'grid'
    root = f"optimize_{mode}"
    root = f"{root}_b_{config['bond_length']}_phi_{config['phi_chromatin']}"
    root = osp.join(dir, root)
    main(root, config, mode)
    # check_all_converged()
