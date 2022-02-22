import os.path as osp

import argparse
import json
import numpy as np
import sys
import csv
from sklearn.metrics.pairwise import polynomial_kernel

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.utils import InteractionConverter, calculate_E_S
from neural_net_utils.argparseSetup import str2int, str2bool, str2float, str2list2D, str2None

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
METHOD_FORMATS={'random':'%d', 'pca':'%.3e', 'pca_split':'%.3e', 'kpca':'%.3e',
                'ground_truth':None, 'k_means':'%d', 'nmf':'%.3e', 'epigenetic':'%d',
                'chromhmm':'%d', 'gnn':'%.3e'}

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--default_config', type=str, default='default_config.json', help='path to default config file')
    parser.add_argument('--ofile', type=str, default='config.json', help='path to output config file')

    # config param arguments
    parser.add_argument('--m', type=int, default=1024, help='number of particles (-1 to infer)')
    parser.add_argument('--k', type=str2int, help='number of particle types (inferred from chi if None)')
    parser.add_argument('--load_configuration_filename', type=str2None, default='input1024.xyz',
                        help='file name of initial config (None to not load)')
    parser.add_argument('--goal_specified', type=str2bool, default=True, help='True will save two lines to chis.txt')
    parser.add_argument('--dump_frequency', type=int, help='set to change dump frequency')
    parser.add_argument('--dump_stats_frequency', type=int, help='set to change dump stats frequency')
    parser.add_argument('--n_sweeps', type=int, help='set to change nSweeps')
    parser.add_argument('--TICG_seed', type=str2int, help='set to change random seed for simulation')
    parser.add_argument('--use_ground_truth_TICG_seed', type=str2bool, help='True to copy seed from config file in sample_folder')
    parser.add_argument('--use_ematrix', type=str2bool, default=False, help='True to use e_matrix')
    parser.add_argument('--use_smatrix', type=str2bool, default=False, help='True to use s_matrix')
    parser.add_argument('--sample_folder', type=str, help='location of sample for ground truth chi')
    parser.add_argument('--relabel', type=str2None,
                        help='specify mark combinations to be relabled (e.g. AB-C will relabel AB mark pairs as mark C)')

    # chi arguments
    parser.add_argument('--use_ground_truth_chi', type=str2bool, default=False, help='True to use ground truth chi and diag chi')
    parser.add_argument('--use_ground_truth_diag_chi', type=str2bool, default=False, help='True to use ground truth diag chi')
    parser.add_argument('--chi_seed', type=str2int, default=None, help='seed for generating chi')

    # diag chi arguments
    parser.add_argument('--diag', type=str2bool, default=False, help='True for diagonal interactions')
    parser.add_argument('--diag_bins', type=str2int, default=20, help='number of diagonal bins')
    parser.add_argument('--max_diag_chi', type=float, default=0.5, help='maximum diag chi value for np.linspace()')

    # plaid chi arguments
    parser.add_argument('--e', type=str2None, help='path to e_matrix to use (.npy or .txt)')
    parser.add_argument('--chi', type=str2list2D, help='chi matrix using latex separator style (if None will chi be generated randomly)')
    parser.add_argument('--save_chi', action="store_true", help='true to save chi to wd')
    parser.add_argument('--save_chi_for_max_ent', action="store_true", help='true to save chi to wd in format needed for max ent')
    parser.add_argument('--min_chi', type=float, default=-1., help='minimum chi value for random generation')
    parser.add_argument('--max_chi', type=float, default=1., help='maximum chi value for random generation')
    parser.add_argument('--fill_diag', type=str2float, help='fill diag of chi with given value (None to skip)')
    parser.add_argument('--fill_offdiag', type=str2float, help='fill off diag of chi with given value (None to skip)')
    parser.add_argument('--ensure_distinguishable', action='store_true', help='true to ensure that corresponding psi is distinguishable')

    args = parser.parse_args()
    args.format = '%.3e' # for writeSeq
    return args

#### x, psi functions ####
def relabel_x_to_psi(x, relabel_str):
    '''
    Relabels seq according to relabel_str.

    Inputs:
        x: m x k np array
        relabel_str: string of format <old>-<new>

    Outputs:
        psi: bead labels np array

    Example:
    consider: <old> = AB, <new> = D, seq is m x 3
    Any particle with both label A and label B, will be relabeled to have
    label D and neither A nor B. Label C will be unaffected.


    If len(<new>) = 1, then LETTERS.find(new) must be >= k
    (i.e label <new> cannot be present in seq already)

    If len(<new>) > 1, then len(<old>) must be 1
    '''
    m, k = x.shape

    old, new = relabel_str.split('-')
    new_labels = [LETTERS.find(i) for i in new]

    old_labels = [LETTERS.find(i) for i in old]
    all_labels = [LETTERS.find(i) for i in old+new]

    if len(new_labels) == 1:
        new_label = new_labels[0]
        assert new_label >= k, "new_label already present"
        psi = np.zeros((m, k+1))
        psi[:, :k] = x

        # find where to assing new_label
        where = np.ones(m) # all True
        for i in old_labels:
            where = np.logical_and(where, x[:, i] == 1)

        # assign new_label
        psi[:, new_label] = where
        # delete old_labels
        for i in old_labels:
            psi[:, i] -= where

        # check that new_label is mutually exclusive from old_labels
        row_sum = np.sum(psi[:, all_labels], axis = 1)
        assert np.all(row_sum <= 1)
    else: # new_label < k
        assert len(old_labels) == 1, "too many old labels"
        old_label = old_labels[0]
        psi = np.delete(seq, old_label, axis = 1)

        for i in new_labels:
            where = np.logical_and(seq[:, i] == 0, seq[:, old_label] == 1)
            psi[:, i] += where

    return psi

def writeSeq(seq, format='%.3e'):
    m, k = seq.shape
    for j in range(k):
        np.savetxt(f'seq{j}.txt', seq[:, j], fmt = format)

# chi functions
def generateRandomChi(args, decimals = 1, rng = np.random.default_rng()):
    '''Initializes random chi array.'''

    # create array with random values in [minval, maxVal]
    rands = rng.random(size=(args.k, args.k)) * (args.max_chi - args.min_chi) + args.min_chi

    # zero lower triangle
    chi = np.triu(rands)

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
    rng = np.random.default_rng(args.chi_seed)
    chi = generateRandomChi(args, rng = rng)
    if args.ensure_distinguishable and args.k < 10: # if k is too large this is too RAM intensive
        conv = InteractionConverter(args.k, chi)
        max_it = 10
        it = 0
        while not uniqueRows(conv.getS()) and it < max_it: # defaults to False
            # generate random chi
            conv.chi = generateRandomChi(args, rng = rng)
            it += 1
        if it == max_it:
            print('Warning: maximum iteration reached')
            print('Warning: particles are not distinguishable')
        chi = conv.chi

    return chi

def set_up_plaid_chi(args, config):
    if args.use_ground_truth_chi:
        args.chi = np.load(osp.join(args.sample_folder, 'chis.npy'))
        print(args.chi)
        _, k = args.chi.shape
        assert k == args.k, f"cols of ground truth chi {args.k} doesn't match cols of seq {k}"
    elif args.chi is None:
        if args.k is not None:
            args.chi = getChis(args)
    elif isinstance(args.chi, str):
        if args.chi == 'nonlinear':
            x = np.load('x.npy') # original particle types that interact nonlinearly
            assert args.k >= 10
            psi = np.zeros((args.m, 15)) # transformation of x such that S = psi \chi psi^T
            psi[:, 0] = (np.sum(x[:, 0:3], axis = 1) == 1) # exactly 1 of A, B, C
            psi[:, 1] = (np.sum(x[:, 0:3], axis = 1) == 2) # exactly 2 of A, B, C
            psi[:, 2] = (np.sum(x[:, 0:3], axis = 1) == 3) # A, B, and C
            psi[:, 3] = x[:, 3] # D
            psi[:, 4] = x[:, 4] # E
            psi[:, 5] = np.logical_and(x[:, 3], x[:, 4]) # D and E
            psi[:, 6] = np.logical_and(x[:, 3], x[:, 5]) # D and F
            psi[:, 7] = np.logical_xor(x[:, 0], x[:, 5]) # either A or F
            psi[:, 8] = x[:, 6] # G
            psi[:, 9] = np.logical_and(np.logical_and(x[:, 6], x[:, 7]), np.logical_not(x[:, 4])) # G and H and not E
            psi[:, 10] = x[:, 7] # H
            psi[:, 11] = x[:, 8] # I
            psi[:, 12] = x[:, 9] # J
            psi[:, 13] = np.logical_or(x[:, 7], x[:, 8]) # H or I
            psi[:, 14] = np.logical_xor(x[:, 8], x[:, 9]) # either I or J

            np.save('psi.npy', psi)
            args.k = 15
            args.chi = np.array([[-1,1.8,-0.5,1.8,0.1,1.3,-0.1,0.1,0.8,1.4,2,1.7,1.5,-0.2,1.1],
                            [0,-1,-0.6,0.6,0.8,-0.8,-0.7,-0.1,0,-0.4,-0.2,0.6,-0.9,1.4,0.3],
                            [0,0,-1,1.6,0,-0.2,-0.4,1.5,0.7,1.8,-0.7,-0.9,0.6,1,0.5],
                            [0,0,0,-1,0.8,1.3,-0.6,0.7,0.1,1.4,0.6,0.7,-0.6,0.5,0.5],
                            [0,0,0,0,-1,0.9,0.2,1.5,1.7,0.1,-0.7,0.8,0.7,1.6,1.6],
                            [0,0,0,0,0,-1,0.6,-0.2,0.8,0.7,-1,-0.9,1.6,0.8,0.3],
                            [0,0,0,0,0,0,-1,-0.2,-0.6,1.8,-0.6,1.9,1.1,0.4,-0.4],
                            [0,0,0,0,0,0,0,-1,1.7,-0.4,1.7,0.2,1.2,1.8,-0.1],
                            [0,0,0,0,0,0,0,0,-1,0.7,0.2,0.8,-0.4,1.4,1.3],
                            [0,0,0,0,0,0,0,0,0,-1,-0.4,0.5,1.9,0.1,0.1],
                            [0,0,0,0,0,0,0,0,0,0,-1,0.9,1,1.3,1],
                            [0,0,0,0,0,0,0,0,0,0,0,-1,1.5,-0.1,0.7],
                            [0,0,0,0,0,0,0,0,0,0,0,0,-1,0.6,-0.6],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0.2],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1]])
        elif args.chi.startswith('polynomial'):
            x = np.load('x.npy') # original particle types that interact nonlinearly
            ind = np.triu_indices(args.k)
            args.k = int(args.k*(args.k+1)/2)
            psi = np.zeros((args.m, args.k))
            for i in range(args.m):
                psi[i] = np.outer(x[i], x[i])[ind]

            np.save('psi.npy', psi)
            args.chi = getChis(args)
    else:
        # zero lower triangle
        args.chi = np.triu(args.chi)
        rows, cols = args.chi.shape
        if args.k is None:
            args.k = rows
        else:
            assert args.k == rows, f'number of particle types {args.k} does not match shape of chi {rows}'
        assert rows == cols, f"chi not square: {args.chi}"
        conv = InteractionConverter(args.k, args.chi)
        if not uniqueRows(conv.getS()):
            print('Warning: particles are not distinguishable')

    # save chi
    if args.chi is not None:
        if args.save_chi:
            print(f'Rank of chi: {np.linalg.matrix_rank(args.chi)}')
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

def set_up_diag_chi(args, config, sample_config):
    if args.use_ground_truth_diag_chi:
        if sample_config is not None:
            args.diag = sample_config["diagonal_on"] # overwrite args
            try:
                chi_diag = sample_config["diag_chis"]
            except KeyError:
                assert args.diag == False
        else:
            print("WARNING: ground truth config missing")
            chi_diag = list(np.linspace(0, args.max_diag_chi, args.diag_bins))
    else:
        chi_diag = list(np.linspace(0, args.max_diag_chi, args.diag_bins))

    config["diagonal_on"] = args.diag
    if args.diag:
        config["diag_chis"] = chi_diag
        if args.save_chi_for_max_ent:
            with open('chis_diag.txt', 'w', newline='') as f:
                wr = csv.writer(f, delimiter = '\t')
                wr.writerow(np.array(config["diag_chis"]))
                wr.writerow(np.array(config["diag_chis"]))

def uniqueRows(array):
    if array is None:
        return False

    if len(np.unique(array, axis=0)) / len(array) == 1.:
        return True
    else:
        return False

def main():
    args = getArgs()
    print(args)

    sample_config = None # ground truth config file
    # used to copy TICG seed and diag chis
    if args.sample_folder is not None:
        sample_config_file = osp.join(args.sample_folder, 'config.json')
        if osp.exists(sample_config_file):
            with open(sample_config_file, 'rb') as f:
                sample_config = json.load(f)


    with open(args.default_config, 'rb') as f:
        config = json.load(f)

    if args.m == -1:
        if osp.exists('x.npy'):
            x = np.load('x.npy')
            args.m, _ = x.shape
        elif osp.exists('e.npy'):
            e = np.load('e.npy')
            args.m, _ = e.shape
        print(f'inferred m = {args.m}')

    if args.relabel is not None:
        x = np.load('x.npy')
        old, new = args.relabel.split('-')
        assert len(new) == 1, f"unsupported relabel: {args.relabel}"
        args.k += 1
        psi = relabel_x_to_psi(x, args.relabel)
        np.save('psi.npy', psi)


    # set up chi
    set_up_plaid_chi(args, config)
    set_up_diag_chi(args, config, sample_config)

    # set up psi
    if osp.exists('psi.npy'):
        psi = np.load('psi.npy')
        print(f'psi loaded with shape {psi.shape}')
    elif osp.exists('x.npy'):
        psi = np.load('x.npy')
        print(f'psi (x) loaded with shape {psi.shape}')
    else:
        psi = None
        print('psi is None')

    # set up e, s
    if psi is not None:
        # psi is only None if not doing max ent
        writeSeq(psi, args.format)

        e, s = calculate_E_S(psi, args.chi)

        if args.use_smatrix:
            np.savetxt('s_matrix.txt', s, fmt='%0.5f')
            np.save('s.npy', s)
        if args.use_ematrix:
            np.savetxt('e_matrix.txt', e, fmt='%0.5f')
            np.save('e.npy', e)
            np.save('s.npy', s) # save s either way
        # if args.m < 2000 and args.k < 10:
        #     print(f'Rank of S: {np.linalg.matrix_rank(s)}')
        #     print(f'Rank of E: {np.linalg.matrix_rank(e)}\n')

    if args.e is not None:
        assert args.use_ematrix
        if osp.exists(args.e):
            e = np.load(args.e)
        else:
            raise Exception(f'e does not exist at {args.e}')
        np.savetxt('e_matrix.txt', e, fmt='%0.5f')
        np.save('e.npy', e)

    if args.use_ematrix or args.use_smatrix:
        config['bead_type_files'] = None
        config["nspecies"] = 0

        # set up config
        if args.use_smatrix:
            assert not args.use_ematrix, 'Cannot use smatrix and ematrix'
            # save smatrix_on
            config['smatrix_on'] = True
            config["smatrix_filename"] = "s_matrix.txt"
        else:
            # save ematrix_on
            config['ematrix_on'] = True
            config["ematrix_filename"] = "e_matrix.txt"
    else:
        # save seq
        config['bead_type_files'] = [f'seq{i}.txt' for i in range(args.k)]

        # save nspecies
        config["nspecies"] = args.k

        # save chi to config
        rows, cols = args.chi.shape
        for row in range(rows):
            for col in range(row, cols):
                key = f'chi{LETTERS[row]}{LETTERS[col]}'
                val = args.chi[row, col]
                config[key] = val

    # save nbeads
    config['nbeads'] = args.m

    # save nSweeps
    if args.n_sweeps is not None:
        config['nSweeps'] = args.n_sweeps

    # save dump frequency
    if args.dump_frequency is not None:
        config['dump_frequency'] = args.dump_frequency
    if args.dump_stats_frequency is not None:
        config['dump_stats_frequency'] = args.dump_stats_frequency

    # save seed
    if args.use_ground_truth_TICG_seed:
        config['seed'] = sample_config['seed']
    elif args.TICG_seed is not None:
        config['seed'] = args.TICG_seed
    else:
        rng = np.random.default_rng()
        config['seed'] = int(rng.integers(1000)) # random int in [0, 1000)

    # save configuration filename
    if args.load_configuration_filename is None:
        config["load_configuration"] = False
        config["load_configuration_filename"] = 'none'
    else:
        config["load_configuration_filename"] = args.load_configuration_filename


    with open(args.ofile, 'w') as f:
        json.dump(config, f, indent = 2)

#### test functions ####
def test():
    args = getArgs()
    args.k = 8
    args.fill_diag = -1
    print(generateRandomChi(args))

    args.fill_diag = None
    args.fill_offdiag = 0
    print(generateRandomChi(args))

def test_nonlinear_chi():
    args = getArgs()
    rng = np.random.default_rng(14)
    args.k = 10
    p_switch = 0.05
    x = np.zeros((args.m, args.k))
    x[0, :] = np.random.choice([1,0], size = args.k)
    for j in range(args.k):
        for i in range(1, args.m):
            if x[i-1, j] == 1:
                x[i, j] = rng.choice([1,0], p=[1 - p_switch, p_switch])
            else:
                x[i, j] = rng.choice([1,0], p=[p_switch, 1 - p_switch])

    psi = np.zeros((args.m, 15))
    psi[:, 0] = (np.sum(x[:, 0:3], axis = 1) == 1) # exactly 1 of A, B, C
    psi[:, 1] = (np.sum(x[:, 0:3], axis = 1) == 2) # exactly 2 of A, B, C
    psi[:, 2] = (np.sum(x[:, 0:3], axis = 1) == 3) # A, B, and C
    psi[:, 3] = x[:, 3] # D
    psi[:, 4] = x[:, 4] # E
    psi[:, 5] = np.logical_and(x[:, 3], x[:, 4]) # D and E
    psi[:, 6] = np.logical_and(x[:, 3], x[:, 5]) # D and F
    psi[:, 7] = np.logical_xor(x[:, 0], x[:, 5]) # either A or F
    psi[:, 8] = x[:, 6] # G
    psi[:, 9] = np.logical_and(np.logical_and(x[:, 6], x[:, 7]), np.logical_not(x[:, 4])) # G and H and not E
    psi[:, 10] = x[:, 7] # H
    psi[:, 11] = x[:, 8] # I
    psi[:, 12] = x[:, 9] # J
    psi[:, 13] = np.logical_or(x[:, 7], x[:, 8]) # H or I
    psi[:, 14] = np.logical_xor(x[:, 8], x[:, 9]) # either I or J

    print(psi)

    args.k = 15
    args.fill_diag = -1
    args.max_chi = 2
    args.min_chi = 0
    chi = getChis(args)
    print(chi)

    s = calculate_S(psi, chi)
    print(s)

def test_polynomial_chi():
    args = getArgs()
    rng = np.random.default_rng(14)
    args.k = 4
    args.m = 5
    p_switch = 0.05
    x = np.zeros((args.m, args.k))
    x[0, :] = np.random.choice([1,0], size = args.k)
    for j in range(args.k):
        for i in range(1, args.m):
            if x[i-1, j] == 1:
                x[i, j] = rng.choice([1,0], p=[1 - p_switch, p_switch])
            else:
                x[i, j] = rng.choice([1,0], p=[p_switch, 1 - p_switch])

    for i in range(args.m):
        print(x[i])
        result = polynomial_kernel(x[i].reshape(-1, 1), x[i].reshape(-1, 1), degree=1, coef0=0)
        print('\t', result)
        result = np.outer(x[i], x[i])
        print('\t', result)

        print(x[i].reshape(-1, 1).T @ x[i].reshape(-1, 1))

    print('---')
    print(x)
    print('--')
    result = polynomial_kernel(x, x, degree=1, coef0=0)
    print('\t', result)


if __name__ == '__main__':
    main()
    # test_nonlinear_chi()
    # test_polynomial_chi()
