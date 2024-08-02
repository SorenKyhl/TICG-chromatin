import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

import pylib.analysis as analysis
from pylib.datapipeline import DataPipeline, get_experiment_marks
from pylib.Maxent import Maxent
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import *
from pylib.utils.load_utils import get_final_max_ent_folder
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_import_log

sys.path.append('/home/erschultz/TICG-chromatin')
import scripts.optimize_grid as optimize_grid
from scripts.data_generation.modify_maxent import get_samples

ROOT = '/home/erschultz'
PROJECT = '/project/depablo/erschultz'
MEDIA = '/media/erschultz/1814ae69-5346-45a6-b219-f77f6739171c/'



def fit():
    dir = osp.join(ROOT, 'chipseq-only', 'sample1')

    y = np.load(osp.join(dir, 'hic.npy')).astype(float)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)

    root = osp.join(dir, f'max_ent6_soren')
    if osp.exists(root):
        return
    os.mkdir(root, mode=0o755)

    # get sequences
    seqs = []
    for i in range(1,7):
        seq = np.loadtxt(f'/home/erschultz/chipseq-only/me-1024-1-encode-chr2_soren/iteration0/pcf{i}.txt')
        seqs.append(seq)
    seqs = np.array(seqs)

    with open('/home/erschultz/chipseq-only/me-1024-1-encode-chr2_soren/resources/config.json') as f:
        config = json.load(f)

    with open('/home/erschultz/chipseq-only/me-1024-1-encode-chr2_soren/resources/params.json') as f:
        params = json.load(f)

    # params['iterations'] = 1
    # params['equilib_sweeps'] = 1000
    params['production_sweeps'] = 300000
    params['stop_at_convergence'] = False
    params['method'] = 'n'

    goals = epilib.get_goals(y, seqs, config) # TODO goals don't match
    params["goals"] = goals

    stdout = sys.stdout
    with open(osp.join(root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(root, params, config, seqs, y, fast_analysis=True,
                    final_it_sweeps=params['production_sweeps'], mkdir=False, bound_diag_chis=False)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout



if __name__ == '__main__':
    fit()
