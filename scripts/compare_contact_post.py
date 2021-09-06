"""
Script for comparing contacts after having already run max ent.
"""
import os
import os.path as osp
import sys

import numpy as np
import argparse

sys.path.insert(1, '/home/erschultz/TICG-chromatin/scripts')
from compare_contact import plotDistanceStratifiedPearsonCorrelation, comparePCA

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--data_folder', type=str, default='../sequences_to_contact_maps/dataset_08_24_21', help='location of input data')
    parser.add_argument('--sample', type=int, default=1201, help='sample id')
    parser.add_argument('--sample_folder', type=str, help='location of input data')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')

    args = parser.parse_args()
    if args.sample_folder is None:
        args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))
    return args

def main():
    args = getArgs()
    y = np.load(osp.join(args.sample_folder, 'y.npy'))[:args.m, :args.m]
    methods = {'PCA', 'PCA_split', 'k_means', 'ground_truth', 'random', 'GNN'}
    # methods = {'random'}

    for file in os.listdir(args.sample_folder):
        if file in methods:
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
                    yhat = np.load(osp.join(file2_path, 'iteration{}'.format(max_it), 'y.npy'))[:args.m, :args.m]
                    # plotDistanceStratifiedPearsonCorrelation(y, yhat, args, dir=file2_path)
                    comparePCA(y, yhat, args, dir=file2_path)


if __name__ == '__main__':
    main()
