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
from neural_net_utils.utils import calculateDistanceStratifiedCorrelation, diagonal_preprocessing, generateDistStats

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    # '../../sequences_to_contact_maps/dataset_04_18_21'
    # "/project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--y', type=str, help='location of input y')
    parser.add_argument('--y_diag_instance', type=str, help='location of input y_diag_instance')
    parser.add_argument('--yhat', type=str, help='location of input yhat')
    parser.add_argument('--yhat_diag_instance', type=str, help='location of input y_diag_instance')
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

    results = np.array(results[1:]).astype(float)
    fig, ax1 = plt.subplots()
    ax1.plot(results[:, 0], results[:, 1], color = 'b')
    ax1.set_ylabel('Accuracy', color = 'b')

    ax2 = ax1.twinx()
    ax2.plot(results[:, 0], results[:, 2], color = 'r')
    ax2.set_ylabel('Pearson R', color = 'r')

    ax1.set_xlabel('Component Index')
    plt.xticks(results[:, 0])
    plt.savefig(osp.join(dir, 'PCA_results.png'))
    plt.close()

# plotting functions
def plotDistanceStratifiedPearsonCorrelation(y, yhat, y_diag, yhat_diag, args, dir = ''):
    n, n = y.shape
    triu_ind = np.triu_indices(n)
    overall_corr_diag, _ = pearsonr(y_diag[triu_ind], yhat_diag[triu_ind])

    overall_corr, corr_arr = calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson')
    avg = np.nanmean(corr_arr)
    title = 'Overall Pearson R: {}'.format(np.round(overall_corr, 3))
    title +='\nAvg Dist Pearson R: {}'.format(np.round(avg, 3))
    title +='\nSCC: {}'.format(np.round(overall_corr_diag, 3))

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
    y = np.load(args.y)[:args.m, :args.m]
    if args.y_diag_instance is not None and osp.exists(args.y_diag_instance):
        y_diag_instance = np.load(args.y_diag_instance)[:args.m, :args.m]
    else:
        meanDist = generateDistStats(y)
        y_diag_instance = diagonal_preprocessing(y, meanDist)

    yhat = np.load(args.yhat)[:args.m, :args.m]
    if args.yhat_diag_instance is not None and osp.exists(args.yhat_diag_instance):
        yhat_diag_instance = np.load(args.yhat_diag_instance)[:args.m, :args.m]
    else:
        meanDist = generateDistStats(yhat)
        y_diag_instance = diagonal_preprocessing(yhat, meanDist)

    plotDistanceStratifiedPearsonCorrelation(y, yhat, y_diag_instance, yhat_diag_instance, args)


if __name__ == '__main__':
    main()
