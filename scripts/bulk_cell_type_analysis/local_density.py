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


def genes():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures/density'
    xyz_list = []
    D_list = []
    m=512
    samples, cell_lines = get_samples('dataset_12_06_23')
    samples = np.array(samples)
    cell_lines = np.array(cell_lines)
    gnn_root = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN690'
    # me_root = 'optimize_grid_b_200_v_8_spheroid_1.5-max_ent10'
    N = len(samples)
    y_arr = np.zeros((N, m, m))
    pc_list = []
    for i, sample in enumerate(samples):
        # print(sample)
        s_dir = osp.join(dir, f'sample{sample}')

        y, y_diag = load_Y(s_dir)
        y_arr[i] = y

        pc1 = epilib.get_pcs(epilib.get_oe(y), 1, normalize=True).reshape(-1)
        pc_list.append(pc1)

        xyz_file = osp.join(s_dir, gnn_root, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps = True, N_min = 5, verbose = False)
        xyz_list.append(xyz)

        D = xyz_to_distance(xyz)
        D_list.append(D)

    result = load_import_log(s_dir)
    start = result['start_mb']
    end = result['end_mb']
    chrom = int(result['chrom'])
    resolution = result['resolution']
    resolution_mb = result['resolution_mb']

    # set up xlabels
    all_labels_float = np.linspace(start, end, m)
    all_labels_int = np.round(all_labels_float, 0).astype(int)
    genome_ticks = [0, m//2, m-1]
    genome_labels = [f'{all_labels_int[i]}' for i in genome_ticks]
    print('genome labels', genome_labels)

    cutoffs = [100, 250, 500, 750]
    X = np.linspace(1, m, m).astype(int)
    for cutoff in cutoffs:
        result_list = []
        for D, cell_line in zip(D_list, cell_lines):
            result_arr = np.zeros(m)
            for i in range(m):
                densities = np.sum(D[:, i, :] < cutoff, axis = 1)
                result = np.mean(densities)
                result_arr[i] = result

            result_list.append(result_arr)

            plt.plot(X, result_arr, label = cell_line)

        plt.xticks(genome_ticks, labels = genome_labels)
        plt.ylabel('Particle Density', fontsize=16)
        plt.xlabel('Genomic Position (Mb)', fontsize=16)
        plt.legend(fontsize=16, loc='center left', bbox_to_anchor = (1,0.5))
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'density_gene_comparison_{cutoff}.png'))
        plt.close()

        # density vs pc1
        pc_arr = np.array(pc_list).flatten()
        result_arr = np.array(result_list).flatten()
        include = result_arr != 0
        plt.scatter(pc_arr[include], result_arr[include])
        plt.ylabel('Particle Density', fontsize=16)
        plt.xlabel('PC1', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'pc1_vs_density_{cutoff}.png'))
        plt.close()

    # pc figure
    for pc, cell_line in zip(pc_list, cell_lines):
        plt.plot(X, pc, label = cell_line)

    plt.xticks(genome_ticks, labels = genome_labels)
    plt.ylabel('PC1', fontsize=16)
    plt.xlabel('Genomic Position (Mb)', fontsize=16)
    plt.legend(fontsize=16, loc='center left', bbox_to_anchor = (1, 0.5))
    plt.tight_layout()
    plt.savefig(osp.join(odir, f'pcs.png'))
    plt.close()

def cell_lines():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures/density/cell_lines'
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
        xyz = xyz_load(xyz_file, multiple_timesteps = True, N_min = 5, verbose = False)

        D = xyz_to_distance(xyz)

        cutoffs = [100, 250, 500, 750]
        X = np.linspace(1, m, m).astype(int)
        for cutoff in cutoffs:
            result_arr = np.zeros(m)
            for j in range(m):
                densities = np.sum(D[:, j, :] < cutoff, axis = 1)
                result = np.mean(densities)
                result_arr[j] = result

            # density vs pc1
            plt.scatter(pc1, result_arr)
            plt.ylabel('Particle Density', fontsize=16)
            plt.xlabel('PC1', fontsize=16)
            plt.tight_layout()
            plt.savefig(osp.join(odir, f'pc1_vs_density_{cutoff}_{cell_line}.png'))
            plt.close()





if __name__ == '__main__':
    # genes()
    cell_lines()
