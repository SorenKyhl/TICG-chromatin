import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from pylib.utils import epilib
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_import_log
from pylib.utils.xyz import xyz_load, xyz_to_distance
from utils import get_samples

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y


def cell_lines():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures/radii'
    m=512
    samples, cell_lines = get_samples('dataset_12_06_23')
    samples = np.array(samples)
    cell_lines = np.array(cell_lines)
    gnn_root = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN690'
    # me_root = 'optimize_grid_b_200_v_8_spheroid_1.5-max_ent10'
    N = len(samples)
    y_arr = np.zeros((N, m, m))
    for i, (sample, cell_line) in enumerate(zip(samples, cell_lines)):
        # print(sample)
        s_dir = osp.join(dir, f'sample{sample}')

        y, y_diag = load_Y(s_dir)
        y_arr[i] = y

        pc1 = epilib.get_pcs(epilib.get_oe(y), 1, normalize=True).reshape(-1)

        xyz_file = osp.join(s_dir, gnn_root, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps = True, N_min = 5, verbose = True)

        # center structures
        for j in range(len(xyz)):
            center = np.nanmean(xyz[j], axis=0)
            xyz[j] -= center

        # compute radii
        radii = np.linalg.norm(xyz, axis = 2)
        print(radii, radii.shape)

        mean_radii = np.mean(radii, axis = 0)
        std_radii = np.std(radii, axis = 0)

        # radii vs position
        X = np.arange(0, m)
        plt.plot(X, mean_radii, c = 'k')
        plt.fill_between(X, mean_radii - std_radii, mean_radii + std_radii, alpha=0.8)
        plt.savefig(osp.join(odir, f'radii_{cell_line}.png'))
        plt.close()

        # density vs pc1
        plt.scatter(pc1, mean_radii)
        plt.ylabel('Radii', fontsize=16)
        plt.xlabel('PC1', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'pc1_vs_radii_{cell_line}.png'))
        plt.close()


if __name__ == '__main__':
    cell_lines()
