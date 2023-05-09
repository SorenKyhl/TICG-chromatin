import os
import os.path as osp
import sys

import numpy as np
import pylib.analysis as analysis
from pylib.Pysim import Pysim
from pylib.utils import default, utils


def run():
    config = utils.load_json('config.json')

    # get sequences
    seqs = np.load('x.npy')

    sim = Pysim('', config, seqs, randomize_seed = False, mkdir = False)

    print('Running Simulation')
    sim.run_eq(10000, config['nSweeps'], 1)

    # with utils.cd(sim.root):
    #     analysis.main_no_maxent()


if __name__ == '__main__':
    run()
