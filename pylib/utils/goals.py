import logging
import math

import numpy as np
from numba import njit


def get_goals(hic, seqs, config, save_path=None):
    """get maximum entropy goals for simulation observables"""
    assert len(seqs.shape) == 2
    if seqs.shape[0] > seqs.shape[1]:
        # need seqs to be kxm
        seqs = seqs.T

    hic /= np.mean(np.diagonal(hic)) # ensure normalized

    plaid = get_goal_plaid(hic, seqs, config)
    diag = get_goal_diag(hic, config)

    if save_path is not None:
        np.savetxt(save_path, plaid, newline=" ", fmt="%.8f")
        np.savetxt(save_path, diag, newline=" ", fmt="%.8f")

    return np.hstack((plaid, diag))

def get_goal_plaid(hic, seqs, config, flat=True, norm=False, adj=True):
    """
    flat: return vector. else return matrix of chis.
    """
    k, n = np.shape(seqs)
    assert n > k, f"seq.shape={seq.shape}"
    assert abs(np.mean(hic.diagonal()) - 1) < 1e-5, f'hic is not normalized: {np.mean(hic.diagonal())} '

    goal_exp = np.zeros((k, k))
    logging.info("getting plaid goals...")
    for i, seqi in enumerate(seqs):
        for j, seqj in enumerate(seqs):
            # goal_exp[i,j] = np.mean((np.outer(seqi,seqj)*hic).flatten())
            goal_exp[i, j] = seqi @ hic @ seqj

            if adj:
                vbead = config["beadvol"]
                vcell = config["grid_size"] ** 3
                goal_exp[i, j] *= (vbead / vcell)

            if norm == "abs":
                goal_exp[i, j] /= np.sum(np.outer(np.abs(seqi), np.abs(seqj)))

            if norm == "n2":
                goal_exp[i, j] /= np.shape(hic)[0] ** 2

            if norm == "n":
                goal_exp[i, j] /= np.shape(hic)[0]

            if norm == "nlogn":
                n = np.shape(hic)[0]
                goal_exp[i, j] /= n * np.log10(n)

            if norm == "n1.5":
                goal_exp[i, j] /= np.shape(hic)[0] ** 1.5

            if norm == "nsqrt":
                n = np.shape(hic)[0]
                f = np.sum(np.outer(np.abs(seqi), np.abs(seqj)))
                goal_exp[i, j] /= f / np.sqrt(n)

    if flat:
        ind = np.triu_indices(k)
        goal_exp = goal_exp[ind]

    return goal_exp

def get_goal_diag(
    hic,
    config,
    prefactors=True,
    get_ps=False
):
    """get goal observables for diagonal interaction

    hic: contact map
    config: simulation configuration
    prefactors: multiply by physical parameters to convert probabilities to simulation observables
    get_ps: return binned p(s) and s vectors, where s is the center of the diagonal bins

    """
    assert abs(np.mean(hic.diagonal()) - 1) < 1e-5, f'hic is not normalized: {np.mean(hic.diagonal())}'

    # check config formatting
    diag_bins = len(config['diag_chis'])
    if 'dense_diagonal_loading' in config:
        loading = config['dense_diagonal_loading']
        config['n_small_bins'] = int(loading * diag_bins)
        assert diag_bins > config['n_small_bins'], f"{diag_bins} < {config['n_small_bins']}"
        config['n_big_bins'] = diag_bins - config['n_small_bins']
    if 'dense_diagonal_cutoff' in config:
        m = len(hic)
        dividing_line = m * config['dense_diagonal_cutoff']
        config['small_binsize'] = int(dividing_line / (config['n_small_bins']))
        config['big_binsize'] = int((m - dividing_line) / config['n_big_bins'])
    if "double_count_main_diagonal" not in config:
        config["double_count_main_diagonal"] = False

    # calculate diag_goal and correction
    goal = []
    correction = []
    for bin in range(diag_bins):
        mask = get_mask(bin, len(hic), config) # mask of hic for given bin
        goal.append(np.sum((mask*hic).flatten()))
        correction.append(np.sum(mask))
    goal = np.array(goal)

    if get_ps:
        ps = goal/correction
        s = diag_bin_centers(config)

    if prefactors:
        vbead = config["beadvol"]
        vcell = config["grid_size"] ** 3
        goal *= vbead / vcell

    if get_ps:
        return goal, ps, s
    else:
        return goal

def get_mask(bin, m, config):
    diag_bins = len(config['diag_chis'])
    dense = config['dense_diagonal_on']
    diag_start = config['diag_start']
    double_count_main_diagonal = config["double_count_main_diagonal"]


    mask = np.zeros((m, m)) # use mask to compute weighted average
    if dense:
        n_small_bins = config['n_small_bins']
        small_binsize = config['small_binsize']
        big_binsize = config['big_binsize']
        dividing_line = n_small_bins * small_binsize
    else:
        binsize = m / diag_bins

    for i in range(m):
        for j in range(i+1):
            d = i-j # don't need abs
            if d < diag_start:
                continue
            if dense:
                if d == 0:
                    curr_bin = 0
                elif d > dividing_line:
                    curr_bin = n_small_bins + math.floor( (d - dividing_line) / big_binsize)
                else:
                    curr_bin =  math.floor( d / small_binsize)
            else:
                curr_bin = int(d/binsize)
            if curr_bin == bin and d >= diag_start:
                mask[i,j] = 1
                mask[j,i] = 1
                if i == j and double_count_main_diagonal:
                    mask[i,j] = 2

    return mask

def diag_bin_centers(config):
    """return centers of diagonal bins"""

    nbeads = config["nbeads"]
    ndiag_bins = len(config["diag_chis"])
    cutoff = config["dense_diagonal_cutoff"]
    loading = config["dense_diagonal_loading"]

    dividing_line = nbeads * cutoff

    n_small_bins = int(loading * ndiag_bins)
    n_big_bins = ndiag_bins - n_small_bins
    small_binsize = int(dividing_line / (n_small_bins))
    big_binsize = int((nbeads - dividing_line) / n_big_bins)


    if config["dense_diagonal_on"]:
        s = np.hstack((np.arange(0, n_small_bins)*small_binsize, np.arange(0,n_big_bins)*big_binsize + dividing_line))
    else:
        s = np.array(range(ndiag_bins))

    return s

def deprecated():
    def get_goal_plaid2(hic, seqs, k, flat=True):
        """
        do we need to get the correct denominator??
        k is the number of sequences (not pcs)
        flat: return vector. else return matrix of chis.
        """

        goal_exp = np.zeros((k, k))
        for i, seqi in enumerate(seqs):
            for j, seqj in enumerate(seqs):
                goal_exp[i, j] = np.mean((np.outer(seqi, seqj) * hic).flatten())
                # correction = np.sum(np.outer(seqi,seqj)) / nbeads**2

        if flat:
            ind = np.triu_indices(k)
            goal_exp = goal_exp[ind]

        return goal_exp


    def mask_diagonal2(contact, bins=16):
        """Returns weighted averages of contact map"""
        rows, cols = contact.shape
        binsize = rows / bins

        assert rows == cols, "contact map must be square"
        nbeads = rows
        measure = []
        correction = []

        for b in range(bins):
            mask = np.zeros_like(contact)
            for r in range(rows):
                for c in range(cols):
                    if int((r - c) / binsize) == b:
                        mask[r, c] = 1
                        mask[c, r] = 1

            measure.append(np.mean((mask * contact).flatten()))
            correction.append(np.sum(mask))

        measure = np.array(measure)
        correction = np.array(correction)
        return measure, correction

    @njit
    def make_mask(
        size, b, cutoff, loading, ndiag_bins, dense_diagonal_on, double_diagonal=True
    ):
        """makes a mask with 1's in subdiagonals inside

        actually faster than numpy version when jitted
        """
        rows, cols = size, size
        mask = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                # if int((r-c)/binsize) == b:
                bin_index = binDiagonal(
                    r, c, cutoff, loading, ndiag_bins, rows, dense_diagonal_on
                )
                if bin_index == b:
                    mask[r, c] = 1
                    mask[c, r] = 1
                    if r == c and double_diagonal:
                        mask[r, r] = 2
        return mask


    @njit
    def make_mask_fast(size, binsize, b):
        # not actually faster if you jit the make_mask code
        rows, cols = size, size
        mask = np.zeros((rows, cols))
        for i in range(binsize):
            mask += np.eye(rows, cols, b * binsize + i)
            mask += np.eye(rows, cols, -b * binsize - i)
        return mask


    @njit
    def mask_diagonal(
        contact, cutoff, loading, ndiag_bins, dense_diagonal_on, double_diagonal
    ):
        """Returns weighted averages of contact map"""
        rows, cols = contact.shape
        # binsize = int(rows/ndiag_bins)

        assert rows == cols, "contact map must be square"
        nbeads = rows
        measure = []
        correction = []

        for b in range(ndiag_bins):
            """
            mask = np.zeros_like(contact)
            for r in range(rows):
                for c in range(cols):
                    if int((r-c)/binsize) == b:
                        mask[r,c] = 1
                        mask[c,r] = 1
                        if r==c:
                            mask[r,r] = 2
            """
            mask = make_mask(
                nbeads, b, cutoff, loading, ndiag_bins, dense_diagonal_on, double_diagonal
            )
            # measure.append(np.mean((mask*contact).flatten()))
            # correction.append(np.sum(mask)/nbeads**2)
            measure.append(np.sum((mask * contact).flatten()))
            correction.append(np.sum(mask))
        """

        for b in range(bins):
            mask = make_mask_fast(rows, cols, binsize, b)
            measure.append(np.mean((mask*contact).flatten()))
            correction.append(np.sum(mask)/nbeads**2)
        """

        measure = np.array(measure)
        correction = np.array(correction)
        return measure, correction


    @njit
    def binDiagonal(i, j, cutoff, loading, ndiag_bins, nbeads, dense_diagonal_on):
        s = abs(i - j)

        if dense_diagonal_on:
            # loading = config["loading"]
            # cutoff = config["cutoff"]
            dividing_line = nbeads * cutoff

            n_small_bins = int(loading * ndiag_bins)
            n_big_bins = ndiag_bins - n_small_bins
            small_binsize = int(dividing_line / (n_small_bins))
            big_binsize = int((nbeads - dividing_line) / n_big_bins)

            if s > dividing_line:
                bin_index = n_small_bins + np.floor((s - dividing_line) / big_binsize)
            else:
                bin_index = np.floor(s / small_binsize)
        else:
            binsize = nbeads / ndiag_bins
            bin_index = np.floor(s / binsize)

        return bin_index
