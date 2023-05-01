import json
import multiprocessing as mp
import os
import os.path as osp
import sys

import numpy as np
import optimize_grid
import pylib.analysis as analysis
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from scripts.get_params import GetSeq


def modify(sample):
    dataset = 'dataset_04_28_23'
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy'))
    nbeads = len(y)

    root = osp.join(dir, 'half_chi1')
    print(root)
    config = utils.load_json(osp.join(dir, 'config.json'))
    config['bead_type_files'] = [f'pcf{i}.txt' for i in range(1, config['nspecies']+1)]
    config['track_contactmap'] = False
    chis = np.array(config['chis'])
    chis[0,0] *= 0.5
    config['chis'] = chis.tolist()
    # config['grid_size'] = 216

    # get sequences
    seqs = np.load(osp.join(dir, 'x.npy'))

    sim = Pysim(root, config, seqs, y, randomize_seed = False, overwrite = True)
    sim.run_eq(10000, 50000, 5)

    with utils.cd(sim.root):
        analysis.main_no_maxent()


def main():
    # with mp.Pool(17) as p:
        # p.map(fit, range(202, 283))
    # for i in range(202, 283):
        # fit(i)
    modify(1)


if __name__ == '__main__':
    # soren()
    # modify_soren()
    main()
