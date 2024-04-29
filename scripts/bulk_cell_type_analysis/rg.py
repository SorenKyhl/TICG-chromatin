import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from pylib.utils import epilib
from pylib.utils.plotting_utils import (RED_CMAP, plot_matrix,
                                        plot_matrix_layout)
from pylib.utils.utils import load_import_log
from pylib.utils.xyz import calculate_rg, xyz_load, xyz_load_cores
from sklearn.cluster import KMeans
from utils import get_samples

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y


def c(y, a, b):
    return a@y@b

def r(y, a, b):
    denom = c(y,a,b)**2
    num = c(y,a,a) * c(y,b,b)
    return num/denom

def compartments():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures'
    xyz_list = []
    m=512
    samples, cell_lines = get_samples('dataset_12_06_23')
    samples = np.array(samples)
    cell_lines = np.array(cell_lines)
    gnn_root = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN631'
    N = len(samples)
    y_arr = np.zeros((N, m, m))
    pca_seq = np.zeros((N, m, 2))
    pca_rab = np.zeros(N)
    for i, sample in enumerate(samples):
        # print(sample)
        s_dir = osp.join(dir, f'sample{sample}')

        y, y_diag = load_Y(s_dir)
        y_arr[i] = y

        seq = epilib.get_pcs(y_diag, 1, normalize = True)[:, 0]
        seq_a = np.zeros_like(seq)
        seq_b = np.zeros_like(seq)
        seq_a[seq > 0] = 1
        seq_b[seq < 0] = 1

        # ensure that positive PC is compartment A based on count of A-A contacts
        # a compartment should have fewer self-self contacts as it is less dense
        count_a = seq_a @ y @ seq_a
        count_b = seq_b @ y @ seq_b
        if count_a < count_b:
            pca_seq[i, :, 0] = seq_a
            pca_seq[i, :, 1] = seq_b
        else:
            pca_seq[i, :, 0] = seq_b
            pca_seq[i, :, 1] = seq_a
        rab = r(y, pca_seq[i, :, 1], pca_seq[i, :, 0])
        pca_rab[i] = rab

        xyz_file = osp.join(s_dir, gnn_root, 'production_out/output.xyz')
        xyz = xyz_load(xyz_file, multiple_timesteps = True, N_min = 5, verbose = False)
        xyz_list.append(xyz)

    plot_n = 12
    rows=2; cols=5
    ind = np.argsort(pca_rab[:plot_n])
    vmin = 0; vmax = np.mean(y_arr)
    plot_matrix_layout(rows, cols, ind,
                        y_arr, pca_rab, cell_lines,
                        RED_CMAP, vmin, vmax,
                        osp.join(odir, f'y_pca_rab.png'))

    window_sizes = [16, 32, 64]
    files = [osp.join(odir, f'rg_w{w}.npy') for w in window_sizes]
    for i, (size, file) in enumerate(zip(window_sizes, files)):
        print(size)
        if osp.exists(file):
            mean_arr = np.load(file)
        else:
            mean_arr = np.zeros((N, m))
            std_arr = np.zeros((N, m))
            X = np.linspace(0, m, m)
            for j, xyz in enumerate(xyz_list):
                left = 0; right = left + size; i=0
                while right <= m:
                    xyz_size = xyz[:, left:right, :]
                    mean, std = calculate_rg(xyz_size)
                    mean_arr[j, i] = mean
                    std_arr[j, i] = std

                    left += 1
                    right += 1
                    i += 1

                plt.plot(X, mean_arr[j], label = j)

            plt.ylabel('Radius of Gyration', fontsize=16)
            plt.xlabel('Genomic Position (bp)', fontsize=16)
            # plt.legend(fontsize=16)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(osp.join(dir,  f'rg_comparison_{size}.png'))
            # plt.show()
            plt.close()

            np.save(file, mean_arr)

        for j in range(2):
            rg_score = np.zeros(N)
            for k in range(N):
                seq = pca_seq[k, :, j]
            rg_score = np.einsum('Nm,m->N', mean_arr, seq)
            rg_score /= m
            print(rg_score)

            ind = np.argsort(rg_score[:plot_n])
            vmin = 0; vmax = np.mean(y_arr)
            plot_matrix_layout(rows, cols, ind,
                                y_arr, rg_score, cell_lines,
                                RED_CMAP, vmin, vmax,
                                osp.join(odir, f'y_rg_score_pc{j}_size{size}.png'))

def genes():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures/rg'
    xyz_list = []
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


    window_sizes = [16, 32, 64]
    for i, size in enumerate(window_sizes):
        X = np.linspace(size, m-size, m).astype(int)

        rg_list = []
        for xyz_i, cell_line in zip(xyz_list, cell_lines):
            mean_arr = np.zeros(m)
            std_arr = np.zeros(m)

            left = 0; right = left + size; i=0
            while right <= m:
                xyz_i_size = xyz_i[:, left:right, :]
                mean, std = calculate_rg(xyz_i_size)
                mean_arr[i] = mean
                std_arr[i] = std

                left += 1
                right += 1
                i += 1

            rg_list.append(mean_arr)

            plt.plot(X, mean_arr, label = cell_line)

        plt.xticks(genome_ticks, labels = genome_labels)
        plt.ylabel('Radius of Gyration', fontsize=16)
        plt.xlabel('Genomic Position (Mb)', fontsize=16)
        plt.legend(fontsize=16, loc='center left', bbox_to_anchor = (1,0.5))
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'rg_gene_comparison_{size}.png'))
        plt.close()

        # rg vs pc1
        pc_arr = np.array(pc_list).flatten()
        rg_arr = np.array(rg_list).flatten()
        include = rg_arr != 0
        plt.scatter(pc_arr[include], rg_arr[include])
        plt.ylabel('Rg', fontsize=16)
        plt.xlabel('PC1', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'pc1_vs_rg_{size}.png'))
        plt.close()

def cell_lines():
    dir = '/home/erschultz/dataset_12_06_23/samples'
    odir = '/home/erschultz/dataset_12_06_23/figures/rg/cell_lines'
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

        window_sizes = [16, 32, 64]
        for i, size in enumerate(window_sizes):
            X = np.linspace(size, m-size, m).astype(int)

            mean_arr = np.zeros(m)
            std_arr = np.zeros(m)

            left = 0; right = left + size; i=0
            while right <= m:
                xyz_size = xyz[:, left:right, :]
                mean, std = calculate_rg(xyz_size)
                mean_arr[i] = mean
                std_arr[i] = std

                left += 1
                right += 1
                i += 1

            # density vs pc1
            include = mean_arr != 0
            plt.scatter(pc1[include], mean_arr[include])
            plt.ylabel('Rg', fontsize=16)
            plt.xlabel('PC1', fontsize=16)
            plt.tight_layout()
            plt.savefig(osp.join(odir, f'pc1_vs_rg_{size}_{cell_line}.png'))
            plt.close()

def soren_20k():
    dir = '/home/erschultz/soren_scaleup/20k-1k/final-20480'
    odir = '/home/erschultz/soren_scaleup/20k-1k/rg20k'
    m = 20480
    y = np.load(osp.join(dir, 'experimental_hic.npy'))

    pc1 = epilib.get_pcs(epilib.get_oe(y), 1, normalize=True).reshape(-1)
    plt.plot(pc1)
    plt.ylabel('PC1', fontsize=16)
    plt.savefig(osp.join(odir, 'pc1.png'))
    plt.close()

    xyz = xyz_load_cores(dir, save = True)

    window_sizes = [1, 2, 4] # num of beads at coarse resolution
    for i, size in enumerate(window_sizes):
        print(size)
        mean_arr = np.zeros(1024)
        std_arr = np.zeros(1024)

        left = 0; right = left + size; i=0
        for i in range(1024):
            if i % 100 == 0:
                print(i)
            # choose left and right such that left:right at 20k reslution is bead i at 1k resolution
            left = i * 20
            right = left + 19

            # add window size
            left_window = max(0, left - size * 10)
            right_window = min(right + size * 10, m)

            xyz_size = xyz[:, left_window:right_window, :]
            mean, std = calculate_rg(xyz_size)
            mean_arr[i] = mean
            std_arr[i] = std

        # density vs pc1
        include = mean_arr != 0
        plt.scatter(pc1[include], mean_arr[include])
        plt.ylabel('Rg', fontsize=16)
        plt.xlabel('PC1', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'pc1_vs_rg_{size}.png'))
        plt.close()


if __name__ == '__main__':
    # compartments()
    # genes()
    cell_lines()
    # soren_20k()
