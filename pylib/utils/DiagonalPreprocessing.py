import os
import os.path as osp

import numpy as np
from scipy.ndimage import uniform_filter


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
        elif stat == 'std':
            np_stat = np.nanstd
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

    def process(y, mean_per_diagonal, triu = False, verbose = False):
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
