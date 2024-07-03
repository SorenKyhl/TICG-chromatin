import copy
import sys
from pathlib import Path
from typing import Callable

import hicrep
import numpy as np
import scipy.ndimage as ndimage
from numba import njit
from skimage.measure import block_reduce

from pylib.utils import default, epilib

"""
collection of functions for manipulating hic maps
"""


def pool(inp, factor, fn=np.nansum, normalize=True, normalize_method='ones'):
    """Resizes input matrix by factor using fn using modified sum pooling

    in modified sum pooling, the sum along the diagonal only includes
    the upper triangle of the pool window. this operation conserves the
    total number of contacts in the contact map
    """
    inp = copy.deepcopy(inp)
    assert len(inp.shape) == 2, f"must be 2d array not {inp.shape}"
    m, _ = inp.shape
    assert m % factor == 0, f"factor must evenly divide m {m}%{factor}={m%factor}"
    # set lower triangle to nan to avoid double counting
    inp[np.tril_indices(m, -1)] = np.nan
    processed = block_reduce(inp, (factor, factor), fn)  # pyright: ignore
    # need to make symmetric again
    processed = np.triu(processed)
    out = processed + np.triu(processed, 1).T

    if normalize:
        out = normalize_hic(out, normalize_method)

    return out

def pool_sum(inp, factor, normalize=True):
    """resizes input matrix by a factor using sum pooling"""
    pooled = block_reduce(inp, (factor, factor), np.nansum)  # pyright: ignore
    if normalize:
        pooled = normalize_hic(pooled)
    return pooled


def pool_diagonal(HiC, normalize=True):
    """downsize matrix by factor using diagonal pooling operation

    only include every factor'th bead
    called "diagonal pooling" because the block_reduce kernel is the identity function
    """
    HiC_new = np.zeros([int(len(HiC) / 2), int(len(HiC) / 2)])
    for i in range(len(HiC_new)):
        for j in range(len(HiC_new)):
            HiC_new[i, j] = HiC[2 * i, 2 * j] + HiC[2 * i + 1, 2 * j + 1]

    if normalize:
        HiC_new = normalize_hic(HiC_new)
    return HiC_new


def pool_double_count(inp, factor: int, fn: Callable = np.nansum):
    inp = copy.deepcopy(inp)
    assert len(inp.shape) == 2, f"must be 2d array not {inp.shape}"
    m, _ = inp.shape
    assert m % factor == 0, f"factor must evenly divide m {m}%{factor}={m%factor}"
    # set lower triangle to nan to avoid double counting
    processed = block_reduce(inp, (factor, factor), fn)  # pyright: ignore
    return processed


def pool_seqs(seqs, factor):
    """downsize sequences by a factor using a mean pooling operation"""

    def pool_seq(seq, factor):
        return block_reduce(seq, (factor), np.mean)  # pyright: ignore

    if seqs.ndim == 1:
        return pool_seq(seqs, factor)
    else:
        return np.array([pool_seq(seq, factor) for seq in seqs])


def unpool_seqs(seqs, factor):
    """upsize sequences by a factor using mean unpooling operation"""
    newseqs = []
    for seq in seqs:
        newseq = []
        for e in seq:
            for i in range(factor):
                newseq.append(e)
        newseqs.append(newseq)
    return np.array(newseqs)


def normalize_hic(hic, method="ones"):
    """divide hic matrix so the mean of the main diagonal is equal to 1"""
    if method == "mean":
        return hic / np.mean(np.diagonal(hic))
    elif method == "ones":
        hic /= np.mean(np.diagonal(hic))
        np.fill_diagonal(hic, 1)
        return hic
    else:
        raise ValueError("method must be 'mean' or 'ones'")


def unpool(inp, factor):
    """upsize matrix by a factor"""
    m, _ = inp.shape

    out = np.zeros((m * factor, m * factor))

    for i, row in enumerate(inp):
        for j, e in enumerate(row):
            # need to deal with main diagonal
            if i == j:
                out[
                    i * factor : i * factor + factor, j * factor : j * factor + factor
                ] = e / (factor * (factor + 1) / 2)
            else:
                out[
                    i * factor : i * factor + factor, j * factor : j * factor + factor
                ] = (e / factor**2)
    return out


def pool_d(hic, factor):
    """downsize hic matrix and return downsized diagonal"""
    x = pool(hic, factor, np.nansum)
    diag = get_diagonal(x)
    x /= max(diag)
    return get_diagonal(x)


def sparsity(x):
    """percent of elements which are zero"""
    return np.sum(x == 0) / len(x) ** 2


def smooth_hic(x, smooth_size=10):
    """gaussian smooth"""
    return ndimage.gaussian_filter(x, (smooth_size, smooth_size))


def load_hic(nbeads, pool_fn=pool_sum, chrom=2, cell="HCT116_auxin"):
    """load hic by pooling preloaded high resolution map"""
    chrom = str(chrom)

    if not default.data_dir.exists():
        default.data_dir.mkdir()

    #hic_file = default.HCT116_hic_20k[chrom]
    hic_file = Path(default.data_dir, f"{cell}_chr{chrom}_20k.npy")

    if not hic_file.exists():
        nbeads_large = 20480
        pipe = copy.deepcopy(default.data_pipeline)
        pipe.set_chrom(chrom)
        pipe.resize(nbeads_large)
        gthic = pipe.load_hic(default.hic_paths[cell])
        np.save(hic_file, gthic)
    else:
        gthic = np.load(hic_file)

    factor = int(len(gthic) / nbeads)
    pooled = pool_fn(gthic, factor)
    return pooled

def load_contact_map(file, chrom = None, resolution = None):
    '''Load contact map from cool, mcool, or npy format.'''
    file_type = file.split('.')[-1]
    if file_type == 'cool':
        clr, binsize = hicrep.utils.readMcool(file, -1)
        if resolution is not None:
            assert resolution == binsize, f"{resolution} != {binsize}"
        if chrom is None:
            y = []
            for chrom in clr.chromnames:
                y.append(clr.matrix(balance=False).fetch(f'{chrom}'))
        else:
            y = clr.matrix(balance=False).fetch(f'{chrom}')
    elif file_type == 'mcool':
        assert resolution is not None
        clr, _ = hicrep.utils.readMcool(file, resolution)
        if chrom is None:
            y = []
            for chrom in clr.chromnames:
                y.append(clr.matrix(balance=False).fetch(f'{chrom}'))
        else:
            y = clr.matrix(balance=False).fetch(f'{chrom}')
    elif file_type == 'npy':
        y = np.load(file)
    else:
        raise Exception(f'Unaccepted file type: {file_type}')

    return y


def load_seqs(nbeads, k, chrom="2", cell="HCT116_auxin"):
    """load sequences by pooling preloaded high resolution sequences"""
    chrom=str(chrom)

    seqs_file = Path(default.data_dir, f"{cell}_chr{chrom}_seqs20k.npy")

    if not seqs_file.exists():
        nbeads_large = 20480
        gthic = load_hic(nbeads_large, chrom=chrom, cell=cell)
        gthic = smooth_hic(gthic)  # this step is very important
        seqs = epilib.get_sequences(gthic, k, randomized=True)
        np.save(seqs_file, seqs)
    else:
        seqs = np.load(seqs_file)
        if k > len(seqs):
            raise ValueError("number of sequences exceeds the loaded amount")
        else:
            seqs = seqs[:k]

    if nbeads < 20480:
        factor = int(seqs.shape[1] / nbeads)
        seqs_final = pool_seqs(seqs, factor)
    else:
        factor = int(nbeads/seqs.shape[1])
        seqs_final = unpool_seqs(seqs, factor)

    return seqs_final


def load_chipseq(nbeads, encode_only=False, chrom=2, return_gthic=False):
    wig_method = "max"
    pipe = copy.deepcopy(default.data_pipeline)

    nbeads_large = 20480
    factor = int(nbeads_large/nbeads)
    pipe.size = nbeads_large
    pipe.res = 5000
    pipe.set_chrom(chrom)

    # need to load gthic to know which indices to drop when loading chip
    gthic = pipe.load_hic(default.HCT116_hic)
    gthic = pool_sum(gthic, factor)
    sequences_total = pipe.load_chipseq_from_directory(default.HCT116_chipseq, wig_method)

    encode_seqs = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3", "H3K9me3"]
    if encode_only:
        sequences = {seq: sequences_total[seq] for seq in encode_seqs}
    else:
        sequences = sequences_total

    chip = {}
    pooled = {}
    for seq in sequences:
        pooled[seq] = pool_seqs(sequences[seq], factor)
        chip[seq] = default.chipseq_pipeline.fit(pooled[seq])

    if return_gthic:
        return np.array(list(chip.values())), gthic
    else:
        return np.array(list(chip.values()))

@njit
def get_diagonal(contact):
    """Returns the probablity of contact as a function of genomic distance"""
    rows, cols = np.shape(contact)
    d = np.zeros(rows)
    for k in range(rows):
        d[k] = np.mean(np.diag(contact, k))
    return d

def rescale_p_s_1(y, target_p):
    y /= np.mean(y.diagonal())
    diag = y.diagonal().copy()
    y_copy = y.copy()
    np.fill_diagonal(y_copy, 0)
    y_copy /= np.mean(y_copy.diagonal(offset=1))
    y_copy *= target_p
    np.fill_diagonal(y_copy, diag)

    return y_copy
