import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from pylib.utils import epilib
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_import_log
from pylib.utils.xyz import xyz_load, xyz_to_distance
from scipy.stats import gaussian_kde
from utils import get_samples

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y


def plot():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures'
    m=512
    samples, cell_lines = get_samples('dataset_12_06_23')
    samples = np.array(samples)
    cell_lines = np.array(cell_lines)
    gnn_root = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN690'
    # me_root = 'optimize_grid_b_200_v_8_spheroid_1.5-max_ent10'
    N = len(samples)
    y_arr = np.zeros((N, m, m))
    for i, sample in enumerate(samples):
        # print(sample)
        s_dir = osp.join(dir, f'sample{sample}')

        y, y_diag = load_Y(s_dir)
        y_arr[i] = y

        pc1 = epilib.get_pcs(epilib.get_oe(y), 1, normalize=True).reshape(-1)

        xyz_file = osp.join(s_dir, gnn_root, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps = True, N_min = 5, verbose = False)

        D = xyz_to_distance(xyz)

        # density vs pc1
        X = y[np.triu_indices(m, 1)][::10]
        Y = np.mean(D, axis=0)[np.triu_indices(m, 1)][::10]

        XY = np.vstack([X,Y])
        Z = gaussian_kde(XY)(XY)
        plt.scatter(Y, X, c=Z)
        plt.xlabel('Distance', fontsize=16)
        plt.ylabel('Contact Frequency', fontsize=16)
        plt.yscale('log')
        # plt.xscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'y_vs_D_{i}.png'))
        plt.close()


if __name__ == '__main__':
    plot()
