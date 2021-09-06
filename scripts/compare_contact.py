import os
import os.path as osp
import sys

import numpy as np
import argparse
import csv

from scipy.stats import spearmanr, pearsonr

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

sys.path.insert(1, '/home/erschultz/sequences_to_contact_maps')
sys.path.insert(1, 'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps')
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

def comparePCA(y, yhat, args, dir = ''):
    # y
    pca_y = PCA()
    pca_y.fit(y)

    # yhat
    pca_yhat = PCA()
    pca_yhat.fit(yhat)

    results = [['Component Index', 'Accuracy', 'Pearson R']]

    for i in range(5):
        comp_y = pca_y.components_[i]
        sign_y = np.sign(comp_y)

        comp_yhat = pca_yhat.components_[i]
        sign_yhat = np.sign(comp_yhat)

        acc = np.sum((sign_yhat == sign_y)) / sign_y.size
        acc = max(acc, 1 - acc)
        corr, pval = pearsonr(comp_yhat, comp_y)
        corr = abs(corr)
        results.append([i, acc, corr])

    with open(osp.join(dir, 'PCA_results.txt'), 'w', newline = '') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerows(results)

# plotting functions
def plotDistanceStratifiedPearsonCorrelation(y, yhat, args, dir = ''):
    overall_corr, corr_arr = calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson')
    avg = np.nanmean(corr_arr)
    title = 'Overall Pearson R: {}'.format(np.round(overall_corr, 3))
    title +='\nAvg Dist Pearson R: {}'.format(np.round(avg, 3))

    plt.plot(np.arange(args.m-2), corr_arr, color = 'black')
    plt.ylim(-0.5, 1)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
    plt.title(title, fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(dir, 'distance_pearson.png'))
    plt.close()

def main():
    args = getArgs()
    y = np.load(args.ifile1)[:args.m, :args.m]
    yhat = np.load(args.ifile2)[:args.m, :args.m]

    plotDistanceStratifiedPearsonCorrelation(y, yhat, args)


if __name__ == '__main__':
    main()
