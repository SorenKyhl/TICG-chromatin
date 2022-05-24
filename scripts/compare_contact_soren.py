'''
Version of compare_contact.py with no dependencies.
Not actively maintained.
'''

import argparse
import csv
import json
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    # '../../sequences_to_contact_maps/dataset_04_18_21'
    # "/project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--y', type=str, help='location of input y')
    parser.add_argument('--y_diag', type=str, help='location of input y_diag')
    parser.add_argument('--yhat', type=str, help='location of input yhat')
    parser.add_argument('--yhat_diag', type=str, help='location of input y_diag')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--dir', type=str, default='', help='location to write to')

    args = parser.parse_args()
    return args

def genomic_distance_statistics(y, mode = 'freq', stat = 'mean'):
    '''
    Calculates statistics of contact frequency/probability as a function of genomic distance
    (i.e. along a give diagonal)

    Inputs:
        mode: freq for frequencies, prob for probabilities
        stat: mean to calculate mean, var for variance

    Outputs:
        result: numpy array where result[d] is the contact frequency/probability stat at distance d
    '''
    if mode == 'prob':
        y = y.copy() / np.max(y)

    if stat == 'mean':
        npStat = np.mean
    elif stat == 'var':
        npStat = np.var
    n = len(y)
    distances = range(0, n, 1)
    result = np.zeros_like(distances).astype(float)
    for d in distances:
        result[d] = npStat(np.diagonal(y, offset = d))

    return result

def calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson'):
    """
    Helper function to calculate correlation stratified by distance.

    Inputs:
        y: target
        yhat: prediction
        mode: pearson or spearman (str)

    Outpus:
        overall_corr: overall correlation
        corr_arr: array of distance stratified correlations
    """
    if mode.lower() == 'pearson':
        stat = pearsonr
    elif mode.lower() == 'spearman':
        stat = spearmanr

    assert len(y.shape) == 2
    n, n = y.shape
    triu_ind = np.triu_indices(n)

    overall_corr, _ = stat(y[triu_ind], yhat[triu_ind])

    corr_arr = np.zeros(n-2)
    corr_arr[0] = np.NaN
    for d in range(1, n-2):
        # n-1, n, and 0 are NaN always, so skip
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, _ = stat(y_diag, yhat_diag)
        corr_arr[d] = corr

    return overall_corr, corr_arr

def diagonal_preprocessing(y, meanDist):
    """
    Removes diagonal effect from contact map y.

    Inputs:
        y: contact map numpy array
        mean: mean contact frequency where mean[dist] is the mean at a given distance

    Outputs:
        result: new contact map
    """
    result = np.zeros_like(y)
    for i in range(len(y)):
        for j in range(i + 1):
            distance = i - j
            exp_d = meanDist[distance]
            if exp_d == 0:
                # this is unlikely to happen
                pass
            else:
                result[i,j] = y[i,j] / exp_d
                result[j,i] = result[i,j]

    return result

def comparePCA(y, yhat, dir):
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
def plotDistanceStratifiedPearsonCorrelation(y, yhat, y_diag, yhat_diag, dir):

    m, _ = y.shape
    triu_ind = np.triu_indices(m)
    overall_corr_diag, _ = pearsonr(y_diag[triu_ind], yhat_diag[triu_ind])

    overall_corr, corr_arr = calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson')
    avg = np.nanmean(corr_arr)

    # save correlations to json
    with open(osp.join(dir, 'distance_pearson.json'), 'w') as f:
        temp_dict = {'overall_pearson': overall_corr,
                     'scc': overall_corr_diag,
                     'avg_dist_pearson': avg}
        json.dump(temp_dict, f)

    # round
    overall_corr_diag = np.round(overall_corr_diag, 3)
    avg = np.round(avg, 3)
    overall_corr = np.round(overall_corr, 3)

    # format title
    title = 'Overall Pearson R: {}'.format(overall_corr)
    title +='\nAvg Dist Pearson R: {}'.format(avg)
    title +='\nSCC: {}'.format(overall_corr_diag)

    plt.plot(np.arange(m-2), corr_arr, color = 'black')
    plt.ylim(-0.5, 1)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
    plt.title(title, fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(dir, 'distance_pearson.png'))
    plt.close()

def main():
    args = getArgs()

    # load y
    if osp.exists(args.y):
        if args.y.endswith('.npy'):
            y = np.load(args.y)
        elif args.y.endswith('.txt'):
            y = np.load_txt(args.y)
        else:
            raise Exception(f'invalid y format {args.y}')
        y = y[:args.m, :args.m] # crop to m
    else:
        raise Exception(f'y does not exist at {args.y}')

    # load y_diag
    if args.y_diag is not None and osp.exists(args.y_diag):
        y_diag = np.load(args.y_diag)[:args.m, :args.m]
    else:
        meanDist = genomic_distance_statistics(y)
        y_diag = diagonal_preprocessing(y, meanDist)

    # load yhat
    yhat = np.load(args.yhat)
    if osp.exists(args.yhat):
        if args.yhat.endswith('.npy'):
            yhat = np.load(args.yhat)
        elif args.yhat.endswith('.txt'):
            yhat = np.load_txt(args.yhat)
        else:
            raise Exception(f'invalid yhat format {args.yhat}')
        yhat = yhat[:args.m, :args.m] # crop to m
    else:
        raise Exception(f'yhat does not exist at {args.yhat}')

    # load yhat_diag
    if args.yhat_diag is not None and osp.exists(args.yhat_diag):
        yhat_diag = np.load(args.yhat_diag)[:args.m, :args.m]
    else:
        meanDist = genomic_distance_statistics(yhat)
        yhat_diag = diagonal_preprocessing(yhat, meanDist)

    plotDistanceStratifiedPearsonCorrelation(y, yhat, y_diag, yhat_diag, args.dir)
    comparePCA(y, yhat, args.dir)


if __name__ == '__main__':
    main()
