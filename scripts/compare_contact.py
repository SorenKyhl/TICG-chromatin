import os
import os.path as osp
import sys

import numpy as np
import argparse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

sys.path.insert(1, '/home/erschultz/sequences_to_contact_maps')
from neural_net_utils.utils import calculateDistanceStratifiedCorrelation

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    # '../../sequences_to_contact_maps/dataset_04_18_21'
    # "/project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--ifile1', type=str, help='location of input data')
    parser.add_argument('--ifile2', type=str, help='location of input data')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')

    args = parser.parse_args()
    return args

# plotting functions
def plotDistanceStratifiedPearsonCorrelation(y, yhat, args):
    overall_corr, corr_arr = calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson')
    title = 'Overall Pearson R: {}'.format(np.round(overall_corr, 3))

    plt.plot(np.arange(args.m-1), corr_arr, color = 'black')
    plt.ylim(-0.5, 1)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
    plt.title(title, fontsize = 16)

    plt.tight_layout()
    plt.savefig('distance_pearson.png')
    plt.close()

def main():
    print(sys.argv)
    args = getArgs()
    y = np.load(args.ifile1)[:args.m, :args.m]
    yhat = np.load(args.ifile2)[:args.m, :args.m]

    plotDistanceStratifiedPearsonCorrelation(y, yhat, args)


if __name__ == '__main__':
    main()
