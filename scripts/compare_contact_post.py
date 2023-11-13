"""
Script for comparing contacts after having already run max ent.
Attempts to iterate through all maxent runs for given smaple_folder
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
from compare_contact import (comparePCA,
                             plotDistanceStratifiedPearsonCorrelation)
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--data_folder', type=str, default='../sequences_to_contact_maps/dataset_08_26_21', help='location of input data')
    parser.add_argument('--sample', type=int, default=1201, help='sample id')
    parser.add_argument('--sample_folder', type=str, help='location of input data')

    args = parser.parse_args()
    if args.sample_folder is None:
        args.sample_folder = osp.join(args.data_folder, 'samples', f'sample{args.sample}')

    print('sample folder:', args.sample_folder)
    return args

def main():
    args = getArgs()
    y, y_diag = load_Y(args.sample_folder)
    for file in os.listdir(args.sample_folder):
        file_path = osp.join(args.sample_folder, file)
        if osp.isdir(file_path):
            for k_file in os.listdir(file_path):
                k_file_path = osp.join(args.sample_folder, file, k_file)
                if osp.isdir(k_file_path):
                    for replicate_file in os.listdir(k_file_path):
                        replicate_file_path = osp.join(k_file_path, replicate_file)
                        if osp.isdir(replicate_file_path) and replicate_file.startswith('replicate'):
                            print(replicate_file_path)
                            yhat = np.load(osp.join(replicate_file_path, 'y.npy')).astype(float)

                            yhat_diag_path = osp.join(replicate_file_path, 'y_diag.npy')
                            if osp.exists(yhat_diag_path):
                                yhat_diag = np.load(yhat_diag_path)
                            else:
                                meanDist = DiagonalPreprocessing.genomic_distance_statistics(yhat)
                                yhat_diag = DiagonalPreprocessing.process(yhat, meanDist)
                                np.save(yhat_diag_path, yhat_diag)

                            plotDistanceStratifiedPearsonCorrelation(y, yhat, y_diag, yhat_diag, dir = replicate_file_path)
                            comparePCA(y, yhat, dir = replicate_file_path)


if __name__ == '__main__':
    main()
