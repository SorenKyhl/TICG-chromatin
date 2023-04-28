import copy
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.ndimage as ndimage
from numba import njit
from skimage.measure import block_reduce

from pylib.utils import default, epilib

"""
collection of functions for manipulating hic maps
"""

class DiagonalPreprocessing():
    '''
    Functions for removing diagonal effect from contact map, y.
    '''
    def get_expected(mean_per_diagonal):
        '''
        Generate matrix of expected contact frequencies.

        Inputs:
            mean_per_diagonal: output of genomic_distance_statistics
        '''
        m = len(mean_per_diagonal)
        expected = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                distance = j - i
                val = mean_per_diagonal[distance]
                expected[i,j] = val
                expected[j,i] = val

        return expected

    def genomic_distance_statistics(y, mode = 'freq', stat = 'mean',
                        zero_diag = False, zero_offset = 1,
                        plot = False, ofile = None, normalize = None,
                        smoothen = False):
        '''
        Calculate statistics of contact frequency/probability as a function of genomic distance
        (i.e. along a give diagonal)

        Inputs:
            mode: freq for frequencies, prob for probabilities
            stat: mean to calculate mean, var for variance
            zero_diag: zero digonals up to zero_offset of contact map
            zero_offset: all diagonals up to zero_offset will be zero-d
            plot: True to plot stat_per_diagonal
            ofile: file path to plot to (None for plt.show())
            normalize: divide each value in stat_per_diagonal by value in normalize
            smoothen: True to apply box filter to contact map

        Outputs:
            stat_per_diagonal: numpy array where result[d] is the contact frequency/probability stat at distance d
        '''
        y = y.copy().astype(np.float64)
        if smoothen:
            y = uniform_filter(y, 3, mode = 'constant')
        if mode == 'prob':
            y /= np.nanmean(np.diagonal(y))

        if zero_diag:
            y = np.triu(y, zero_offset + 1)
        else:
            y = np.triu(y)

        if stat == 'mean':
            np_stat = np.nanmean
        elif stat == 'var':
            np_stat = np.nanvar
        m = len(y)
        distances = range(0, m, 1)
        stat_per_diagonal = np.zeros_like(distances).astype(float)
        for d in distances:
            stat_per_diagonal[d] = np_stat(np.diagonal(y, offset = d))


        if isinstance(normalize, np.ndarray) or isinstance(normalize, float):
            stat_per_diagonal = np.divide(stat_per_diagonal, normalize)

        if plot:
            plt.plot(stat_per_diagonal)
            if ofile is not None:
                plt.savefig(ofile)
            else:
                plt.show()
            plt.close()

        return np.array(stat_per_diagonal)

    def process(y, mean_per_diagonal, triu = False, verbose = True):
        """
        Inputs:
            y: contact map numpy array or path to .npy file
            mean_per_diagonal: mean contact frequency distribution where
                mean_per_diagonal[distance] is the mean at a given distance
            triu: True if y is 1d array of upper triangle instead of full 2d contact map
                (relatively slow version)

        Outputs:
            result: new contact map
        """
        if isinstance(y, str):
            assert osp.exists(y)
            y = np.load(y)
        y = y.astype('float64')
        if triu:
            y = triu_to_full(y)

        for d in range(len(mean_per_diagonal)):
            expected = mean_per_diagonal[d]
            if expected == 0 and verbose:
                # this is unlikely to happen
                print(f'WARNING: 0 contacts expected at distance {d}')

        m = len(mean_per_diagonal)
        result = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1):
                distance = i - j
                expected = mean_per_diagonal[distance]
                if expected != 0:
                    result[i,j] = y[i,j] / expected
                    result[j,i] = result[i,j]

        if triu:
            result = result[np.triu_indices(m)]

        return result

    def process_chunk(dir, mean_per_diagonal, odir = None, chunk_size = 100,
                    jobs = 1, sparse_format = False, verbose = True):
        '''
        Faster version of process when using many contact maps
        but RAM is limited.

        Assumes contact maps are in triu format (i.e. an N x m x m array of N m by m
        contact maps has been reshaped to size N x m*(m+1)/2 )
        '''
        zeros = np.argwhere(mean_per_diagonal == 0)
        if len(zeros) > 0:
            if verbose:
                print(f'WARNING: 0 contacts expected at distance {zeros}')
            # replace with minimum observed mean
            mean_per_diagonal[zeros] = np.min(mean_per_diagonal[mean_per_diagonal > 0])

        expected = DiagonalPreprocessing.get_expected(mean_per_diagonal)

        # infer m
        m = len(mean_per_diagonal)

        N = len([f for f in os.listdir(dir) if f.endswith('.npy')])
        files = [f'y_sc_{i}.npy' for i in range(N)]
        if odir is None:
            sc_contacts_diag = np.zeros((N, int(m*(m+1)/2)))
        for i in range(0, N, chunk_size):
            # load sc contacts
            sc_contacts = np.zeros((len(files[i:i + chunk_size]), int(m*(m+1)/2)))
            for j, file in enumerate(files[i:i + chunk_size]):
                sc_contacts[j] = np.load(osp.join(dir, file))

            # process
            result = DiagonalPreprocessing.process_bulk(sc_contacts,
                                                    expected = expected,
                                                    triu = True)
            if odir is None:
                sc_contacts_diag[i:i+chunk_size] = result
            else:
                if jobs > 1:
                    mapping = []
                    for y, f in zip(result, files[i:i + chunk_size]):
                        mapping.append((osp.join(odir, f), y))
                    with multiprocessing.Pool(jobs) as p:
                        p.starmap(np.save, mapping)
                else:
                    for y, f in zip(result, files[i:i + chunk_size]):
                        np.save(osp.join(odir, f), y)


        if odir is None:
            if sparse_format:
                sc_contacts_diag = sp.csr_array(sc_contacts_diag)

            return sc_contacts_diag

    def process_bulk(y_arr, mean_per_diagonal = None, expected = None,
                    triu = False):
        '''Faster version of process when using many contact maps.'''
        y_arr = y_arr.astype('float64', copy = False)

        if expected is None:
            assert mean_per_diagonal is not None
            zeros = np.argwhere(mean_per_diagonal == 0)
            if len(zeros) > 0:
                print(f'WARNING: 0 contacts expected at distance {zeros}')
                # replace with minimum observed mean
                mean_per_diagonal[zeros] = np.min(mean_per_diagonal[mean_per_diagonal > 0])

            expected = DiagonalPreprocessing.get_expected(mean_per_diagonal)

        if triu:
            m, _ = expected.shape
            expected = expected[np.triu_indices(m)]
        if sp.issparse(y_arr):
            result = y_arr.copy()
            result.data /= np.take(expected, result.indices)
        else:
            result = y_arr / expected

        return result



def pool(inp, factor, fn=np.nansum, normalize=True):
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
        out = normalize_hic(out)

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
