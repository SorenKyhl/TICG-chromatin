import argparse
import json
import numpy as np
import sys
import csv

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--default_config', type=str, default='default_config.json', help='path to default config file')
    parser.add_argument('--ofile', type=str, default='config.json', help='path to output config file')

    # config param arguments
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--k', type=int, help='number of particle types (inferred from chi if None)')
    parser.add_argument('--load_configuration_filename', type=str, default='input1024.xyz', help='file name of initial config')
    parser.add_argument('--goal_specified', type=int, default=1, help='1=true, will save two lines to chis.txt')
    parser.add_argument('--dump_frequency', type=int, help='set to change dump frequency')
    parser.add_argument('--dump_stats_frequency', type=int, help='set to change dump stats frequency')
    parser.add_argument('--nSweeps', type=int, help='set to change nSweeps')
    parser.add_argument('--seed', type=int, help='set to change random seed')

    # diag chi arguments
    parser.add_argument('--diag', type=str2bool, default=False, help='True for diagonal interactions')
    parser.add_argument('--max_diag_chi', type=float, default=0.5, help='maximum diag chi value for np.linspace()')

    # chi arguments
    parser.add_argument('--chi', type=str2list, help='chi matrix using latex separator style (if None will be generated randomly)')
    parser.add_argument('--save_chi', action="store_true", help='true to save chi to wd')
    parser.add_argument('--save_chi_for_max_ent', action="store_true", help='true to save chi to wd in format needed for max ent')
    parser.add_argument('--min_chi', type=float, default=-1., help='minimum chi value for random generation')
    parser.add_argument('--max_chi', type=float, default=1., help='maximum chi value for random generation')
    parser.add_argument('--fill_diag', type=float, help='fill diag of chi with given value (None to skip)')
    parser.add_argument('--fill_offdiag', type=float, help='fill off diag of chi with given value (None to skip)')
    parser.add_argument('--ensure_distinguishable', action='store_true', help='true to ensure that corresponding psi is distinguishable')


    args = parser.parse_args()
    return args

def str2list(v, sep1 = '\\', sep2 = '&'):
    """
    Helper function for argparser, converts str to list by splitting on sep1, then on sep2.

    Example for sep1 = '\\', sep2 = '&': "i & j \\ k & l" -> [[i, j], [k, l]]

    Inputs:
        v: string (any spaces will be ignored)
        sep: separator
    """
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() == 'none':
            return None
        else:
            v = v.replace(' ', '') # get rid of spaces
            result = [i.split(sep2) for i in v.split(sep1)]
            result = np.array(result, dtype=float)
            return result
    else:
        raise argparse.ArgumentTypeError('str value expected.')

def str2bool(v):
    """
    Helper function for argparser, converts str to boolean for various string inputs.
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Inputs:
        v: string
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def generateRandomChi(args, decimals = 1):
    '''Initializes random chi array.'''
    # create array with random values in [minval, maxVal]
    rands = np.random.rand(args.k, args.k) * (args.max_chi - args.min_chi) + args.min_chi

    # make symmetric chi array
    chi = np.tril(rands) + np.triu(rands.T, 1)

    if args.fill_offdiag is not None:
        # fills off diag chis with value of fill_offdiag
        chi_diag = np.diagonal(chi)
        chi = np.ones((args.k, args.k)) * args.fill_offdiag
        di = np.diag_indices(args.k)
        chi[di] = chi_diag
    if args.fill_diag is not None:
        # fills diag chis with value of fill_diag
        di = np.diag_indices(args.k)
        chi[di] = args.fill_diag

    return np.round(chi, decimals = decimals)

def getChis(args):
    conv = InteractionConverter(args.k, generateRandomChi(args))
    if args.ensure_distinguishable:
        max_it = 10
        it = 0
        while not conv.PsiUniqueRows() and it < max_it: # defaults to False
            # generate random chi
            conv.chi = generateRandomChi(args)
            conv.updatePsi()
            it += 1
        if it == max_it:
            print('Warning: maximum iteration reached')
            print('Warning: particles are not distinguishable')

    return conv.chi

class InteractionConverter():
    """Class that allows conversion between epigenetic mark bit string pairs and integer type id"""
    def __init__(self, k, chi = None):
        self.k = k
        self.chi = chi
        self.allStrings = self.generateAllBinaryStrings()
        if chi is not None:
            self.Psi = self.allStrings @ self.chi @ self.allStrings.T
        else:
            self.Psi = None

    def fillArrayWithAllBinaryStrings(self, n, arr, temp_arr, i, row = 0):
        # https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/
        if i == n:
            arr.append(temp_arr.copy())
            row += 1
            return row

        # First assign "1" at ith position
        # and try for all other permutations
        # for remaining positions
        temp_arr[i] = 1
        self.fillArrayWithAllBinaryStrings(n, arr, temp_arr, i + 1, row)

        # And then assign "0" at ith position
        # and try for all other permutations
        # for remaining positions
        temp_arr[i] = 0
        self.fillArrayWithAllBinaryStrings(n, arr, temp_arr, i + 1, row)

    def generateAllBinaryStrings(self):
        arr = []
        temp_arr = [None]*self.k
        self.fillArrayWithAllBinaryStrings(self.k, arr, temp_arr, 0)
        np_arr = np.array(arr).astype(np.int8)
        return np_arr

    def PsiUniqueRows(self):
        if self.Psi is None:
            return False

        print("Chi:\n", self.chi)
        print("Net interaction matrix:\n", self.Psi)

        if len(np.unique(self.Psi, axis=0)) / len(self.Psi) == 1.:
            return True
        else:
            return False

    def updatePsi(self):
        assert self.chi is not None, "set chi first"
        self.Psi = self.allStrings @ self.chi @ self.allStrings.T

def main():
    args = getArgs()

    with open(args.default_config, 'rb') as f:
        config = json.load(f)

    # process chi
    if args.chi is None:
        assert args.k is not None, "chi and k cannot both be None"
        args.chi = getChis(args)
    else:
        # chi is not None
        rows, cols = args.chi.shape
        if args.k is None:
            args.k = rows
        else:
            assert args.k == rows, 'number of particle types does not match shape of chi'
        assert rows == cols, "chi not square: {}".format(args.chi)
        assert np.allclose(args.chi, args.chi.T), "chi is not symmetric: {}".format(args.chi)
        conv = InteractionConverter(args.k, args.chi)
        if not conv.PsiUniqueRows():
            print('Warning: particles are not distinguishable')

    # save chi to wd
    if args.save_chi:
        np.savetxt('chis.txt', args.chi, fmt='%0.5f')
        np.save('chis.npy', args.chi)
    elif args.save_chi_for_max_ent:
        with open('chis.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow(args.chi[np.triu_indices(args.k)])
            if args.goal_specified:
                wr.writerow(args.chi[np.triu_indices(args.k)])
            else:
                # get random chi
                chi = getChis(args)
                wr.writerow(chi[np.triu_indices(args.k)])
        with open('chis_diag.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow(np.array(config["diag_chis"]))
            wr.writerow(np.array(config["diag_chis"]))


    # save chi to config
    rows, cols = args.chi.shape
    for row in range(rows):
        for col in range(cols):
            key = 'chi{}{}'.format(LETTERS[row], LETTERS[col])
            val = args.chi[row, col]
            config[key] = val

    # set up diag chis
    config["diagonal_on"] = args.diag
    if args.diag:
        chi_diag = np.linspace(0, args.max_diag_chi, 20)
        config["diag_chis"] = list(chi_diag)

    # save nbeads
    config['nbeads'] = args.m

    # save nspecies
    config["nspecies"] = args.k

    # save nSweeps
    if args.nSweeps is not None:
        config['nSweeps'] = args.nSweeps

    # save dump frequency
    if args.dump_frequency is not None:
        config['dump_frequency'] = args.dump_frequency
    if args.dump_stats_frequency is not None:
        config['dump_stats_frequency'] = args.dump_stats_frequency

    # save seed
    if args.seed is not None:
        config['seed'] = args.seed

    # save configuration filename
    config["load_configuration_filename"] = args.load_configuration_filename

    # save chipseq files
    config['chipseq_files'] = ['seq{}.txt'.format(i) for i in range(args.k)]

    with open(args.ofile, 'w') as f:
        json.dump(config, f, indent = 2)

def test():
    args = getArgs()
    args.k = 8
    args.fill_diag = -1
    print(generateRandomChi(args))

    args.fill_diag = None
    args.fill_offdiag = 0
    print(generateRandomChi(args))

def test2():
    conv = InteractionConverter(2, np.array([[-1, 2],[2, -1]]))
    print(conv.allStrings)
    print(conv.PsiUniqueRows())

if __name__ == '__main__':
    main()
