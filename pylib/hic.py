import numpy as np
import copy
from skimage.measure import block_reduce

from pylib import epilib


"""
collection of functions for manipulating hic maps
"""

def pool(inp, factor, fn=np.nansum, normalize=False):
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
        out = main_diagonal_to_one(out)

    return out

def pool_double_count(inp, factor, fn=np.nansum):
    inp = copy.deepcopy(inp)
    assert len(inp.shape) == 2, f'must be 2d array not {inp.shape}'
    m, _ = inp.shape
    assert m % factor == 0, f'factor must evenly divide m {m}%{factor}={m%factor}'
    # set lower triangle to nan to avoid double counting
    processed = block_reduce(inp, (factor, factor), fn)
    return processed


def main_diagonal_to_one(hic):
    d = epilib.get_diagonal(hic)
    return hic/d[0]

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
