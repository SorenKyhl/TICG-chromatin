"""
Script for comparing contacts after having already run max ent.
"""
import os
import os.path as osp
import sys

import matplotlib
import numpy as np
import argparse

from scripts.contact_map import plotContactMap
from scripts.compare_contact import plotDistanceStratifiedPearsonCorrelation, comparePCA
from scripts.makeLatexTable import METHODS

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/Research/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.utils import diagonal_preprocessing
from data_summary_plots import genomic_distance_statistics


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--data_folder', type=str, default='../sequences_to_contact_maps/dataset_08_26_21', help='location of input data')
    parser.add_argument('--sample', type=int, default=1201, help='sample id')
    parser.add_argument('--sample_folder', type=str, help='location of input data')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')

    args = parser.parse_args()
    if args.sample_folder is None:
        args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))
        print(args.sample_folder)
    return args

def main():
    args = getArgs()
    y = np.load(osp.join(args.sample_folder, 'y.npy'))[:args.m, :args.m]
    y_diag_instance = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))[:args.m, :args.m]
    plotContactMap(y_diag_instance, ofile = osp.join(args.sample_folder, 'y_diag_instance.png'), vmax = 'max')
    v_max = np.max(y_diag_instance)


    for file in os.listdir(args.sample_folder):
        if file in METHODS:
            for file2 in os.listdir(osp.join(args.sample_folder, file)):
                file2_path = osp.join(args.sample_folder, file, file2)
                if osp.isdir(file2_path):
                    print(file2_path)
                    # find max it
                    max_it = -1
                    for file3 in os.listdir(file2_path):
                        if file3.startswith('iteration'):
                            it = int(file3[9:])
                            if it > max_it:
                                max_it = it
                    max_it_path = osp.join(file2_path, 'iteration{}'.format(max_it))
                    yhat = np.load(osp.join(max_it_path, 'y.npy'))[:args.m, :args.m].astype(float)
                    yhat_diag_path = osp.join(max_it_path, 'y_diag_instance.npy')
                    if osp.exists(yhat_diag_path):
                        yhat_diag_instance = np.load(yhat_diag_path)
                    else:
                        meanDist = genomic_distance_statistics(yhat)
                        yhat_diag_instance = diagonal_preprocessing(yhat, meanDist)
                    dif = yhat_diag_instance - y_diag_instance
                    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                     [(0, 'blue'),
                                                     (0.5, 'white'),
                                                      (1, 'red')], N=126)
                    plotContactMap(dif, ofile = osp.join(max_it_path, 'dif.png'), vmin = 'min', vmax = 'max', cmap = cmap)
                    plotContactMap(yhat_diag_instance, ofile = osp.join(max_it_path, 'y_diag_instance.png'), vmax = v_max)
                    plotDistanceStratifiedPearsonCorrelation(y, yhat, y_diag_instance, yhat_diag_instance, args, dir=file2_path)
                    comparePCA(y, yhat, dir=file2_path)


if __name__ == '__main__':
    main()
