
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
import matplotlib.pyplot as plt

def xyz_to_distance(xyz, verbose = False, aggregate=True):
    """calculate mean pairwise distances from xyz file"""
    N, m, _ = xyz.shape
    D = np.zeros((N, m, m), dtype = np.float32)
    for i in range(N):
        if verbose:
            print(i)
        D_i = nan_euclidean_distances(xyz[i])
        D[i] = D_i

    if aggregate:
        return np.mean(D, axis=0)
    else:
        return D

def get_ds(D, plot = True):
    """mean parwise distance as a function of genomic separation, s"""
    if len(D.shape) == 3:
        D = np.nanmean(D, axis=0)

    d_s = np.zeros(len(D))
    for i in range(len(D)):
        d_s[i] = np.nanmean(np.diagonal(D, i))
    
    if plot:
        plot_ds(d_s)
    return d_s

def plot_ds(d_s):
    plt.plot(d_s, '-o')
    plt.xlabel("s") 
    plt.ylabel("distance [nm]")

def abhist(seqs, D, pc = 0):
    """
    plot histogram of pairwise distances conditioned on the 
    sign of the principal component
    """
    mask = [bool(x) for x in seqs[pc]>0]
    squaremask = np.outer(mask,mask)

    neg = [bool(x) for x in seqs[pc]<0]
    negmask = np.outer(neg, neg)
    pos_d = D_mean[squaremask]
    neg_d = D_mean[negmask]
    
    neither_mask = np.outer(neg,mask)+np.outer(mask,neg)
    neither = D_mean[neither_mask]
    
    mean_pos_d = np.mean(pos_d)
    mean_neg_d = np.mean(neg_d)
    mean_neither = np.mean(neither)
    
    plt.hist(pos_d, bins=100, alpha=0.25, label=f"A-A: PC{pc+1}+, mean: {mean_pos_d:.0f} nm");
    plt.hist(neg_d, bins=100, alpha=0.25, label=f"B-B: PC{pc+1}-, mean: {mean_neg_d:.0f} nm");
    plt.hist(neither, bins=100, alpha=0.25, label=f"A-B, mean: {mean_neither:.0f} nm");

    plt.xlabel("distance [nm]")
    plt.ylabel("frequency")
    plt.legend()
    plt.title(f"A/B region distance distributions (PC{pc+1})")
