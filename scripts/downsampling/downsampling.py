import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from pylib import analysis
from pylib.Pysim import Pysim
from pylib.utils import utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import plot_matrix, plot_mean_dist

sys.path.append('/home/erschultz')

from sequences_to_contact_maps.scripts.load_utils import \
    get_final_max_ent_folder


def run_long_simulation():
    if not osp.exists('/home/erschultz/downsampling_analysis/'):
        os.mkdir('/home/erschultz/downsampling_analysis')
    root = '/home/erschultz/downsampling_analysis/long_simulation'

    # dir2 = '/home/erschultz/dataset_02_04_23/samples/sample212/optimize_grid_b_140_phi_0.03-max_ent'
    dir = '/home/erschultz/dataset_04_28_23/samples/sample324'
    config = utils.load_json(osp.join(dir, 'config.json'))
    config['bead_type_files'] = [f'pcf{i}.txt' for i in range(1, config['nspecies']+1)]
    config['track_contactmap'] = True
    config['dump_frequency'] = 1000
    # k = config['nspecies']
    # chis = np.zeros((k, k))
    # LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # for i in range(k):
    #     chis[i,i] = config[f'chi{LETTERS[i]}{LETTERS[i]}']
    # config['chis'] = chis.tolist()
    # config['profiling_on'] = False

    # get sequences
    seqs = np.load(osp.join(dir, 'x.npy'))

    sim = Pysim(root, config, seqs, None, randomize_seed = False, overwrite = True)
    sim.run_eq(50000, 300000, 1)

    with utils.cd(sim.root):
        analysis.main_no_maxent()

def split_long_simulation():
    dir = '/home/erschultz/downsampling_analysis'
    production_dir = osp.join(dir, 'long_simulation3/production_out')
    for i in [1/3]:
    # [1, 2, 3, 4, 5, 10, 25, 50, 75, 100]:
        y = np.loadtxt(osp.join(production_dir, f'contacts{int(300000 * i/100)}.txt'))
        print(i)
        odir = osp.join(dir, f'sample{i}')
        if not osp.exists(odir):
            os.mkdir(odir)
        np.save(osp.join(odir, 'y.npy'), y)
        plot_matrix(y, osp.join(odir, 'y.png'), vmax = 'mean')


def analysis():
    dir = '/home/erschultz/downsampling_analysis/samples3'
    GNN_ID = 403
    gnn_scc = []
    max_ent_scc = []
    samples = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100]
    for i in samples:
        s_dir = osp.join(dir, f'sample{i}')
        # max_ent_dir = osp.join(s_dir, 'optimize_grid_b_140_phi_0.03-max_ent')
        # final = get_final_max_ent_folder(max_ent_dir)
        # dist_pearson = utils.load_json(osp.join(final, 'distance_pearson.json'))
        # max_ent_scc.append(dist_pearson['scc_var'])

        gnn_dir = osp.join(s_dir, f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}')
        dist_pearson = utils.load_json(osp.join(gnn_dir, 'distance_pearson.json'))
        gnn_scc.append(dist_pearson['scc_var'])


    print(gnn_scc)
    # plt.plot(samples, max_ent_scc, label='Max Ent', color='blue')
    plt.plot(samples, gnn_scc, label='GNN', color='red')
    plt.xlabel('Downsampling %', fontsize=16)
    plt.ylabel('SCC', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/erschultz/downsampling_analysis/scc.png')

if __name__ == '__main__':
    # run_long_simulation()
    split_long_simulation()
    # analysis()
