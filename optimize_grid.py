import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pylib.utils.epilib as epilib
from pylib.optimize import get_bonded_simulation_xyz, optimize_config
from pylib.Pysim import Pysim
from pylib.utils import default, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import plot_matrix, plot_mean_dist
from pylib.utils.xyz import (xyz_load, xyz_to_angles, xyz_to_contact_grid,
                             xyz_to_distance)
from scipy import optimize
from scipy.stats import norm
from scripts.contact_map import plot_max_ent
from scripts.data_generation.modify_maxent import simple_histogram
from sklearn.metrics import mean_squared_error


def run(dir, config):
    print(dir)
    sim = Pysim(dir, config, None, randomize_seed = False, overwrite = True)
    sim.run_eq(10000, config['nSweeps'], 1)

def bonded_simulations():
    dataset = osp.join(default.root, 'dataset_bonded')
    if not osp.exists(dataset):
        os.mkdir(dataset, mode=0o755)

    base_config = default.bonded_config
    base_config['beadvol'] = 130000
    base_config['seed'] = 1
    base_config["nSweeps"] = 350000
    base_config["dump_frequency"] = 1000
    base_config["dump_stats_frequency"] = 100
    base_config['grid_size'] = 1000

    mapping = []
    for boundary_type, ar in [('spherical', 1.0), ('spheroid', 1.5), ('spheroid', 2.0)]:
        boundary_dir = f'boundary_{boundary_type}'
        if ar != 1.0:
            boundary_dir += f'_{ar}'
        boundary_dir = osp.join(dataset, boundary_dir)
        if not osp.exists(boundary_dir):
            os.mkdir(boundary_dir, mode=0o755)
        for bond_type in ['gaussian']:
            bond_dir = osp.join(boundary_dir, f'bond_type_{bond_type}')
            if not osp.exists(bond_dir):
                os.mkdir(bond_dir, mode=0o755)
            for m in [512]:
                m_dir = osp.join(bond_dir, f'm_{m}')
                if not osp.exists(m_dir):
                    os.mkdir(m_dir, mode=0o755)
                for b in [160, 180, 200]:
                    b_dir = osp.join(m_dir, f'bond_length_{b}')
                    if not osp.exists(b_dir):
                        os.mkdir(b_dir, mode=0o755)
                    for v in [7, 9]:
                        v_dir = osp.join(b_dir, f'v_{v}')
                        if not osp.exists(v_dir):
                            os.mkdir(v_dir, mode=0o755)
                        for k_angle in [0]:
                            if bond_type == 'DSS'and k_angle != 0:
                                continue
                            for theta_0 in [180]:
                                if theta_0 == 180:
                                    k_angle_dir = osp.join(v_dir, f'angle_{k_angle}')
                                elif k_angle == 0:
                                    continue
                                else:
                                    k_angle_dir = osp.join(v_dir, f'angle_{k_angle}_theta0_{theta_0}')

                                if not osp.exists(k_angle_dir):
                                    os.mkdir(k_angle_dir, mode=0o755)

                                config = base_config.copy()
                                config['bond_length'] = b
                                config['target_volume'] = v
                                config['nbeads'] = m
                                config["bond_type"] = bond_type
                                config['boundary_type'] = boundary_type
                                config['aspect_ratio'] = ar
                                if k_angle != 0:
                                    config['angles_on'] = True
                                    config['k_angle'] = k_angle
                                    config['theta_0'] = theta_0
                                mapping.append((k_angle_dir, config))

    print(len(mapping))
    with mp.Pool(min(len(mapping), 16)) as p:
        p.starmap(run, mapping)

def plot_bond_length():
    dataset = osp.join(default.root, 'dataset_bonded')
    X = []
    Y = []
    for boundary_type, ar in [('spherical', 1.0)]:
        if boundary_type == 'spheroid':
            boundary_type += f'_{ar}'
        for bond_type in ['gaussian']:
            for m in [512]:
                for b in [261]:
                    for phi in [0.03]:
                        for k_angle in [0]:
                            for theta_0 in [180]:
                                dir = osp.join(dataset, f'boundary_{boundary_type}/bond_type_{bond_type}/m_{m}/bond_length_{b}/phi_{phi}/angle_{k_angle}')
                                if theta_0 != 180:
                                    dir += f'_theta0_{theta_0}'
                                if not osp.exists(dir):
                                    print(f"{dir} does not exist")
                                    continue

                                xyz = xyz_load(osp.join(dir, 'production_out/output.xyz'),
                                                multiple_timesteps=True, N_min=1)
                                xyz = xyz.reshape(-1, m, 3)
                                D = xyz_to_distance(xyz)
                                N = len(D)
                                data = []
                                for i in range(N):
                                    data.append(np.diagonal(D[i], 1))
                                simple_histogram(data, xlabel='Bond Length', odir=dir,
                                                ofname='bond_length_dist.png', dist=norm)
                                #
                                angles = xyz_to_angles(xyz).flatten()
                                simple_histogram(angles, xlabel=r'Angle $\theta$',
                                                odir=dir,
                                                ofname='angle_dist.png', dist=norm)

                                log_labels = np.linspace(0, (m-1), m)
                                D = np.nanmean(D, axis = 0)
                                meanDist_D = DiagonalPreprocessing.genomic_distance_statistics(D, mode='freq')
                                plt.plot(log_labels, meanDist_D, label='Simulation')

                                D_exp = np.load('/home/erschultz/Su2020/samples/sample1013/D_crop.npy')
                                meanDist_D_exp = DiagonalPreprocessing.genomic_distance_statistics(D_exp, mode='freq')
                                nan_rows = np.isnan(meanDist_D_exp)
                                plt.plot(log_labels[~nan_rows], meanDist_D_exp[~nan_rows],
                                            label='Experiment', color='k')
                                plt.legend()
                                plt.ylabel('Distance (nm)')
                                plt.xlabel('Polymer Distance (m)')
                                plt.xscale('log')
                                plt.tight_layout()
                                plt.savefig(osp.join(dir, 'meanDist_D.png'))
                                plt.close()

        #                         with open(osp.join(k_angle_dir, 'production_out/log.log')) as f:
        #                             for line in f.readlines():
        #                                 if line.startswith('simulation volume'):
        #                                     vol = line.split(' ')[-2]
        #                                     print(vol)
        #                                     break
        #                         X.append(phi)
                                X.append(k_angle)
                                Y.append(np.sqrt(np.mean(data)))
    #
    #
    # plt.plot(X, Y)
    # # plt.axhline(b, ls='--', c='k')
    # # plt.xlabel('Simulation Volume um^3')
    # plt.xlabel('k_angle')
    # plt.ylabel('Root Mean Square Bond Length')
    # plt.xscale('log')
    # plt.savefig('/home/erschultz/TICG-chromatin/figures/bond_length2.png')
    # plt.close()

def main(root, config, mode='grid_angle10'):
    gthic = np.load(osp.join(osp.split(root)[0], 'y.npy')).astype(float)
    config['nbeads'] = len(gthic)
    # config["nSweeps"] = 100000
    # config["dump_frequency"] = 100
    # config["dump_stats_frequency"] = 100

    if mode.startswith('grid') or mode.startswith('distance'):
        optimum = optimize_config(config, gthic, mode, 0.5, 2.0, root)
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
                        ref_label = 'Reference',  label = f'Bonded {mode} optimal',
                        color = 'b', title = title)
        plot_mean_dist(p_s_sim, root, 'mean_dist_log.png',
                        None, True, ref = p_s_exp,
                        ref_label = 'Reference',  label = f'Bonded {mode} optimal',
                        color = 'b', title = title)

        print(f"optimal {mode} is: {optimum}")
        with open(osp.join(root, f'{mode}.txt'), 'w') as f:
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
    bonded_simulations()
    # test_optimize()
    # plot_bond_length()
