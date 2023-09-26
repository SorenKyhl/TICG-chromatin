import argparse
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import numpy as np
import pylib.analysis as analysis
from pylib.Pysim import Pysim
from pylib.utils import default, utils

sys.path.append('/home/erschultz/TICG-chromatin')
import scripts.get_config as get_config
import scripts.get_params_old as get_params
from scripts.contact_map import plot_all


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser',
                                    fromfile_prefix_chars='@',
                                    allow_abbrev = False)
    parser.add_argument('--start', type=int,
                        help='first sample')
    parser.add_argument('--end', type=int,
                        help='last sample')
    parser.add_argument('--jobs', type=int,
                        help='number of jobs')

    parser.add_argument('--scratch', type=str,
                        help='absolute path to scratch')
    parser.add_argument('--data_folder', type=str,
                        help='absolute path to dataset')
    parser.add_argument('--overwrite', action='store_true',
                        help='True to overwrite')
    parser.add_argument('--m', type=int,
                        help='num particles')

    args, _ = parser.parse_known_args()
    return args


def check_dir(odir, overwrite):
    if osp.exists(odir):
        if overwrite:
            print(f'Overwriting {odir}')
            shutil.rmtree(odir)
        else:
            raise Exception(f'Output directory already exists: {odir}')


def run(args, i):
    odir = osp.join(args.data_folder, f'samples/sample{i}')
    odir_scratch = osp.join(args.scratch, f'{i}')
    os.mkdir(odir_scratch, mode=0o755)
    os.chdir(odir_scratch)

    defaults = '/home/erschultz/TICG-chromatin/defaults'
    shutil.copyfile(osp.join(defaults, 'config_erschultz.json'),
                    osp.join(odir_scratch, 'default_config.json'))
    check_dir(odir, args.overwrite)

    args_file = osp.join(args.data_folder, f'setup/sample_{i}.txt')


    stdout = sys.stdout
    with open(osp.join(odir_scratch, 'params.log'), 'w') as sys.stdout:
        get_params.main(args_file)
    sys.stdout = stdout

    stdout = sys.stdout
    with open(osp.join(odir_scratch, 'config.log'), 'w') as sys.stdout:
        get_config.main(args_file)
    sys.stdout = stdout

    stdout = sys.stdout
    with open(osp.join(odir_scratch, 'log.log'), 'w') as sys.stdout:
        config = utils.load_json('config.json')

        # get sequences
        if osp.exists('x.npy'):
            seqs = np.load('x.npy')
        else:
            seqs = None

        sim = Pysim('', config, seqs, randomize_seed = False, mkdir = False)

        print('Running Simulation')
        sim.run_eq(10000, config['nSweeps'], 1)
    sys.stdout = stdout

    stdout = sys.stdout
    with open(osp.join(odir_scratch, 'contact_map.log'), 'w') as sys.stdout:
        args.save_npy = True
        args.random_mode = True
        args.plot = True
        args.save_folder = ''
        args.sample_folder = ''
        plot_all(args)
    sys.stdout = stdout

    os.remove('default_config.json')
    shutil.move(odir_scratch, odir)

def run_wrapper():
    args = getArgs()
    mapping = []
    for i in range(args.start, args.end+1):
        mapping.append((args, i))
    with mp.Pool(args.jobs) as p:
        p.starmap(run, mapping)

if __name__ == '__main__':
    run_wrapper()
