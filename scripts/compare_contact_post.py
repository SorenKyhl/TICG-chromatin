"""
Script for comparing contacts after having already run max ent.
"""
import argparse
import os
import os.path as osp

import numpy as np

from ..sequences_to_contact_maps.scripts.data_summary_plots import \
    genomic_distance_statistics
from ..sequences_to_contact_maps.scripts.utils import diagonal_preprocessing
from .compare_contact import (comparePCA,
                              plotDistanceStratifiedPearsonCorrelation)


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
    y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))[:args.m, :args.m]

    for file in os.listdir(args.sample_folder):
        file_path = osp.join(args.sample_folder, file)
        if osp.isdir(file_path) and file.startswith('ground'):
            for k_file in os.listdir(file_path):
                k_file_path = osp.join(args.sample_folder, file, k_file)
                if osp.isdir(k_file_path):
                    for replicate_file in os.listdir(k_file_path):
                        replicate_file_path = osp.join(k_file_path, replicate_file)
                        if osp.isdir(replicate_file_path):
                            print(replicate_file_path)
                            yhat = np.load(osp.join(replicate_file_path, 'y.npy')).astype(float)

                            yhat_diag_path = osp.join(replicate_file_path, 'y_diag.npy')
                            if osp.exists(yhat_diag_path):
                                yhat_diag = np.load(yhat_diag_path)
                            else:
                                meanDist = genomic_distance_statistics(yhat)
                                yhat_diag= diagonal_preprocessing(yhat, meanDist)

                            plotDistanceStratifiedPearsonCorrelation(y, yhat, y_diag, yhat_diag, dir = replicate_file_path)
                            comparePCA(y, yhat, dir = replicate_file_path)


if __name__ == '__main__':
    main()
