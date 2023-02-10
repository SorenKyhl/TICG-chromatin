import numpy as np
import copy
from skimage.measure import block_reduce
import scipy.ndimage as ndimage


from pylib import epilib


"""
collection of functions for manipulating hic maps
"""

def pool(inp, factor, fn=np.nansum, normalize=True):
    """
    Resizes input matrix by factor using fn.
    if inp is 1024x1024 and factor=2, out is 512x512
    """
    inp = copy.deepcopy(inp)
    assert len(inp.shape) == 2, f'must be 2d array not {inp.shape}'
    m, _ = inp.shape
    assert m % factor == 0, f'factor must evenly divide m {m}%{factor}={m%factor}'
    # set lower triangle to nan to avoid double counting
    inp[np.tril_indices(m, -1)] = np.nan
    processed = block_reduce(inp, (factor, factor), fn)
    # need to make symmetric again
    processed = np.triu(processed)
    out = processed + np.triu(processed, 1).T

    if normalize:
        out = normalize_hic(out)

    return out

def pool_sum(inp, factor, normalize=True): 
    pooled = block_reduce(inp, (factor,factor), np.nansum)
    if normalize:
        pooled = normalize_hic(pooled)
    return pooled


def pool_diagonal(HiC, normalize=True):
    """
    reduce size of matrix by factor,
    only include every factor'th bead
    called "diagonal pooling" because the block_reduce kernel is the identity function
    """
    HiC_new = np.zeros([int(len(HiC)/2), int(len(HiC)/2)])
    for i in range(len(HiC_new)):
        for j in range(len(HiC_new)):
            HiC_new[i,j] = (HiC[2*i, 2*j]+HiC[2*i+1,2*j+1])

    if normalize:
        HiC_new = normalize_hic(HiC_new)
    return HiC_new

def pool_double_count(inp, factor, fn=np.nansum):
    inp = copy.deepcopy(inp)
    assert len(inp.shape) == 2, f'must be 2d array not {inp.shape}'
    m, _ = inp.shape
    assert m % factor == 0, f'factor must evenly divide m {m}%{factor}={m%factor}'
    # set lower triangle to nan to avoid double counting
    processed = block_reduce(inp, (factor, factor), fn)
    return processed

def pool_seqs(seqs, factor):
    def pool_seq(seq, factor):
        return block_reduce(seq, (factor), np.mean) 

    if seqs.ndim == 1:
        return pool_seq(seqs, factor)
    else:
        return np.array([pool_seq(seq, factor) for seq in seqs])

def unpool_seqs(seqs, factor):
    newseqs = []
    for seq in seqs:
        newseq = []
        for e in seq:
            for i in range(factor):
                newseq.append(e)
        newseqs.append(newseq)
    return np.array(newseqs)

def normalize_hic(hic):
    return hic / np.mean(np.diagonal(hic))

def unpool(inp, factor):
    """
    Increases input matrix by a factor
    """
    m, _ = inp.shape
    
    out = np.zeros((m*factor, m*factor))
    
    for i, row in enumerate(inp):
        for j, e in enumerate(row):
            # need to deal with main diagonal
            if i == j:
                out[i*factor:i*factor+factor, j*factor:j*factor+factor] = e / (factor*(factor+1)/2)
            else:
                out[i*factor:i*factor+factor, j*factor:j*factor+factor] = e / factor**2
    return out

def pool_d(hic, factor):
    """
    pools hic matrix, returns pooled diagonal
    """
    x = pool(hic, factor, np.nansum)
    diag = ep.get_diagonal(x)
    x /= max(diag)
    return ep.get_diagonal(x)

def sparsity(x):
    return np.sum(x==0)/len(x)**2

def smooth_hic(x, smooth_size=10):
    return ndimage.gaussian_filter(x,(smooth_size,smooth_size))
