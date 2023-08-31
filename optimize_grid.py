import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import numpy as np
import pylib.utils.epilib as epilib
from pylib.optimize import get_bonded_simulation_xyz, optimize_config
from pylib.Pysim import Pysim
from pylib.utils import default, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import plot_matrix, plot_mean_dist
from pylib.utils.xyz import xyz_load, xyz_to_contact_grid
from scipy import optimize
from scripts.contact_map import plot_max_ent
from sklearn.metrics import mean_squared_error


def run(dir, config):
    sim = Pysim(dir, config, None, randomize_seed = False, mkdir = False)
    sim.run_eq(10000, config['nSweeps'], 1)

def bonded_simulations():
    dataset = osp.join(default.root, 'dataset_bonded')
    if not osp.exists(dataset):
        os.mkdir(dataset, mode=0o755)

    config = default.bonded_config
    config['beadvol'] = 130000
    config['seed'] = 1
    config["nSweeps"] = 350000
    config["dump_frequency"] = 1000
    config["dump_stats_frequency"] = 100


    mapping = []
    for m in [512]:
        m_dir = osp.join(dataset, f'm_{m}')
        if not osp.exists(m_dir):
            os.mkdir(m_dir, mode=0o755)
        for b in [140, 261]:
            b_dir = osp.join(m_dir, f'bond_length_{b}')
            if not osp.exists(b_dir):
                os.mkdir(b_dir, mode=0o755)
            for phi in [0.01, 0.03]:
                phi_dir = osp.join(b_dir, f'phi_{phi}')
                if not osp.exists(phi_dir):
                    os.mkdir(phi_dir, mode=0o755)

                config = config.copy()
                config['bond_length'] = b
                config['phi_chromatin'] = phi
                config['nbeads'] = m
                print(phi_dir, config['phi_chromatin'])
                mapping.append((phi_dir, config))

    with mp.Pool(min(len(mapping), 10)) as p:
        p.starmap(run, mapping)


def main(root, config, mode='grid_angle10'):

    gthic = np.load(osp.join(osp.split(root)[0], 'y.npy')).astype(float)
    config['nbeads'] = len(gthic)
    # config["nSweeps"] = 100000
    # config["dump_frequency"] = 100
    # config["dump_stats_frequency"] = 100

    if mode.startswith('grid'):
        optimum = optimize_config(config, gthic, 'grid', 0.8, 2.0, root)
        p_s_exp = DiagonalPreprocessing.genomic_distance_statistics(gthic, 'prob')
        xyz = get_bonded_simulation_xyz(config)
        y = xyz_to_contact_grid(xyz, optimum, dtype=float)
        np.save(osp.join(root, 'y.npy'), y)
        plot_matrix(y, osp.join(root, 'y.png'))
        p_s_exp = DiagonalPreprocessing.genomic_distance_statistics(gthic, 'prob')
        p_s_sim = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        rmse = mean_squared_error(p_s_sim, p_s_exp, squared = False)
        diff = np.abs(p_s_exp[1] - p_s_sim[1])
        title = f'RMSE: {np.round(rmse, 5)}'
        plot_mean_dist(p_s_sim, root, 'mean_dist.png',
                        None, False, ref = p_s_exp,
                        ref_label = 'Reference',  label = 'Bonded Grid Optimal',
                        color = 'b', title = title)
        plot_mean_dist(p_s_sim, root, 'mean_dist_log.png',
                        None, True, ref = p_s_exp,
                        ref_label = 'Reference',  label = 'Bonded Grid Optimal',
                        color = 'b', title = title)

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
        optimum = optimize_config(config, gthic, 'angle', 0.0, 2.0, root, f'neighbor_{s}', mkdir=False)
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

def test_optimize():
    y = np.load('/home/erschultz/Su2020/samples/sample1004/y.npy')
    dir = '/home/erschultz/Su2020/samples/sample1004/optimize_grid_b_261_phi_0.01 (copy)'
    y_sim = np.load(osp.join(dir, 'y.npy'))
    with open(osp.join(dir, 'config.json')) as f:
        config = json.load(f)
    xyz = get_bonded_simulation_xyz(config)
    # xyz = xyz_load(osp.join(dir, 'iteration6/output.xyz'))
    optimum = np.loadtxt(osp.join(dir, 'grid_size.txt'))
    print(optimum)
    y_xyz = xyz_to_contact_grid(xyz, optimum, dtype=float)

    p_s_exp = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    p_s_sim = DiagonalPreprocessing.genomic_distance_statistics(y_sim, 'prob')
    p_s_sim_xyz = DiagonalPreprocessing.genomic_distance_statistics(y_xyz, 'prob')

    plot_mean_dist(p_s_exp, dir, 'mean_dist_test_log.png',
                    None, True,
                    ref = p_s_sim_xyz, ref_label = 'output.xyz',  ref_color = 'g',
                    ref2 = p_s_sim, ref2_label = 'y.npy', ref2_color = 'b',
                    label = 'Experiment', color = 'k')



if __name__ == "__main__":
    # check_all_converged()
    # bonded_simulations()
    test_optimize()
