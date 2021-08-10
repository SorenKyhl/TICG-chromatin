import argparse
import json
import numpy as np

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--chi', type=str2list, help='chi matrix using latex separator style (if None will be generated randomly)')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--k', type=int, help='number of particle types (inferred from chi if None)')
    parser.add_argument('--save_chi', action="store_true", help='true to save chi to wd')
    parser.add_argument('--min_chi', type=float, default=-1., help='minimum chi value for random generation')
    parser.add_argument('--max_chi', type=float, default=1., help='maximum chi value for random generation')

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
            print(v)
            result = [i.split(sep2) for i in v.split(sep1)]
            print(result)
            result = np.array(result, dtype=float)
            print(result)
            return result
    else:
        raise argparse.ArgumentTypeError('str value expected.')

def generateRandomChi(n_types, only_diag = False, minVal = -1., maxVal = 1., decimals = 1):
    '''Initializes random chi array.'''
    # create array with random values in [minval, maxVal]
    rands = np.random.rand(n_types, n_types) * (maxVal - minVal) + minVal

    # make symmetric chi array
    chi = np.tril(rands) + np.triu(rands.T, 1)

    if only_diag:
        # only diag mode sets all off diagonal elements to 0
        # i.e. type i only interacts with type i
        chi_diag = np.diagonal(chi)
        chi = np.zeros((n_types, n_types))
        di = np.diag_indices(n_types)
        chi[di] = chi_diag

    return np.round(chi, decimals = decimals)

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
    with open('default_config.json', 'rb') as f:
        config = json.load(f)

    args = getArgs()

    # process chi
    if args.chi is None:
        assert args.k is not None, "chi and k cannot both be None"
        conv = InteractionConverter(args.k)
        while not conv.PsiUniqueRows(): # defaults to False
            # generate random chi
            conv.chi = generateRandomChi(args.k, minVal = args.min_chi, maxVal = args.max_chi)
            conv.updatePsi()
        args.chi = conv.chi
    else:
        print("Chi:\n", args.chi)
        rows, cols = args.chi.shape
        if args.k is None:
            args.k = rows
        else:
            assert args.k == rows, 'number of particle types does not match shape of chi'
        assert rows == cols, "chi not square".format(args.chi)
        conv = InteractionConverter(args.k, args.chi)
        if not conv.PsiUniqueRows():
            print('Warning: particles are not distinguishable')

    # save chi to wd
    if args.save_chi:
        np.savetxt('chis.txt', args.chi, fmt='%0.5f')
        np.save('chis.npy', args.chi)

    # save chi to config
    letters='ABCDEFG'
    rows, cols = args.chi.shape
    for row in range(rows):
        for col in range(cols):
            key = 'chi{}{}'.format(letters[row], letters[col])
            val = args.chi[row, col]
            config[key] = val

    # save nbeads
    config['nbeads']=args.m

    # save chipseq files
    config['chipseq_files'] = ['seq{}.txt'.format(i) for i in range(args.k)]

    with open('config.json', 'w') as f:
        json.dump(config, f, indent = 2)

if __name__ == '__main__':
    main()
