import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from pylib.utils import epilib
from pylib.utils.plotting_utils import (RED_CMAP, plot_matrix,
                                        plot_matrix_layout)
from pylib.utils.utils import load_import_log
from pylib.utils.xyz import calculate_rg, xyz_load, xyz_load_cores
from utils import load_gene_loci, load_rpkm


def main():
    gene_rpkm_dict = load_rpkm()
    m=512
    odir = '/home/erschultz/dataset_HCT116_RAD21_KO/rg'
    gnn_root = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN690_xyz'

    dir = '/home/erschultz/dataset_HCT116_RAD21_KO/samples/sample1'

    import_log = load_import_log(dir)
    gene_loci_dict = load_gene_loci(import_log, pad=5000000)
    genes = gene_loci_dict.keys()

    xyz_file = osp.join(dir, gnn_root, 'production_out/output.xyz')
    xyz = xyz_load(xyz_file, multiple_timesteps = True, N_min = 5, verbose = True)

    window_sizes = [0, 4, 8, 16, 32, 64]
    for i, size in enumerate(window_sizes):
        rg = []
        expression = []
        for gene in genes:
            if gene in gene_rpkm_dict.keys():
                expression.append(gene_rpkm_dict[gene])
                left, right = gene_loci_dict[gene]
                left = left - size//2
                right = right + size//2 + 1
                xyz_size = xyz[:, left:right, :]
                mean, std = calculate_rg(xyz_size)
                rg.append(mean)

        # expression vs pc1
        plt.scatter(expression, rg)
        plt.ylabel('Rg', fontsize=16)
        plt.xlabel('Expression', fontsize=16)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'expression_vs_rg_{size}.png'))
        plt.close()

if __name__ == '__main__':
    main()
