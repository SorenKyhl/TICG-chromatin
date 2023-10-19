import argparse
import csv
import json
import os.path as osp
import sys

import numpy as np
from pylib.utils.energy_utils import calculate_D, calculate_diag_chi_step
from pylib.utils.plotting_utils import plot_matrix
from sklearn.metrics.pairwise import polynomial_kernel

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import ArgparserConverter
from sequences_to_contact_maps.scripts.utils import LETTERS, crop


def getArgs(args_file=None, args_tmp=None):
    parser = argparse.ArgumentParser(description='Base parser', fromfile_prefix_chars='@',
                                allow_abbrev = False)
    AC = ArgparserConverter()

    parser.add_argument('--config_ifile', type=str, default='config.json',
                        help='path to default config file')
    parser.add_argument('--config_ofile', type=str, default='config.json',
                            help='path to output config file')
    parser.add_argument('--args_file', type=AC.str2None,
                        help='file with more args to load')

    # config params
    parser.add_argument('--m', type=int, default=-1,
                        help='number of particles (-1 to infer)')
    parser.add_argument('--load_configuration_filename', type=AC.str2None,
                        help='file name of initial config (None to not load)')
    parser.add_argument('--dump_frequency', type=int,
                        help='set to change dump frequency')
    parser.add_argument('--dump_stats_frequency', type=int,
                        help='set to change dump stats frequency')
    parser.add_argument('--n_sweeps', type=int,
                        help='set to change nSweeps')
    parser.add_argument('--TICG_seed', type=AC.str2int,
                        help='set to change random seed for simulation (None for random)')
    parser.add_argument('--use_ground_truth_TICG_seed', type=AC.str2bool,
                        help='True to copy seed from config file in sample_folder')
    parser.add_argument('--sample_folder', type=str,
                        help='location of sample for ground truth chi')
    parser.add_argument('--bond_type', type=AC.str2None, default='gaussian',
                        help='type of bonded interaction')
    parser.add_argument('--parallel', type=AC.str2bool, default=False,
                        help='True to run simulation in parallel')
    parser.add_argument('--num_threads', type=int, default=2,
                        help='Number of threads if parallel is True')
    parser.add_argument('--phi_chromatin', type=AC.str2float,
                        help='chromatin volume fraction')
    parser.add_argument('--volume', type=int,
                        help='simulation volume')
    parser.add_argument('--bond_length', type=float, default=16.5,
                        help='bond length')
    parser.add_argument('--k_angle', type=float, default=0.0,
                        help='k for angle term')
    parser.add_argument('--boundary_type', type=str, default='spherical',
                        help='simulation boundary type {cubic, spherical}')
    parser.add_argument('--aspect_ratio', type=float, default=1.0,
                        help='for simulation boundary type = spheroid')
    parser.add_argument('--track_contactmap', type=AC.str2bool, default=False,
                        help='True to dump contact map every dump_frequency')
    parser.add_argument('--gridmove_on', type=AC.str2bool, default=True,
                        help='True to use grid MC move')
    parser.add_argument('--grid_size',
                        help='TICG grid size (None to load from optimize_grid)')
    parser.add_argument('--bead_vol', type=float, default=520,
                        help='bead volume')
    parser.add_argument('--update_contacts_distance', type=AC.str2bool, default=False,
                        help='True to use distance instead of grid')

    # chi config params
    parser.add_argument('--use_ground_truth_chi', type=AC.str2bool, default=False,
                        help='True to use ground truth chi and diag chi')

    # constant chi config params
    parser.add_argument('--constant_chi', type=float, default=0,
                        help='constant chi parameter between all beads')

    # energy config params
    parser.add_argument('--use_smatrix', type=AC.str2bool, default=True,
                        help='True to use s_matrix')
    parser.add_argument('--use_dmatrix', type=AC.str2bool, default=True,
                        help='True to use d_matrix')
    parser.add_argument('--use_lmatrix', type=AC.str2bool, default=True,
                        help='True to use l_matrix')

    parser.add_argument('--e_constant', type=float, default=0,
                        help='constant to add to e')
    parser.add_argument('--s_constant', type=float, default=0,
                        help='constant to add to s')

    # diagonal config params
    parser.add_argument('--diag_pseudobeads_on', type=AC.str2bool, default=True)
    parser.add_argument('--dense_diagonal_on', type=AC.str2bool, default=False,
                        help='True to place 1/2 of beads left of cutoff')


    # max_ent options
    parser.add_argument('--max_ent', action="store_true",
                        help='true to save chi to wd in format needed for max ent')
    parser.add_argument('--mode', type=str,
                        help='mode for max_ent')


    args, _ = parser.parse_known_args()
    if args.args_file is None and args_file is not None:
        args.args_file = args_file
    if args.args_file is not None:
        assert osp.exists(args.args_file), f'{args.args_file} does not exist'
        print(f'parsing {args.args_file}')
        argv = sys.argv.copy()
        print(f'argv: {argv}\n')
        argv.append(f'@{args.args_file}') # appending means args_file will override other args
        print(f'argv: {argv}\n')
        args, unknown = parser.parse_known_args(argv)
        print(f'unknown: {unknown}\n')

    if args_tmp is not None:
        # hacky solution
        args.n_sweeps = args_tmp.n_sweeps
        args.dump_frequency = args_tmp.dump_frequency
        args.TICG_seed = args_tmp.TICG_seed
        args.phi_chromatin = args_tmp.phi_chromatin
        args.bead_vol = args_tmp.bead_vol
        args.bond_length = args_tmp.bond_length
        args.track_contactmap = args_tmp.track_contactmap

    return args

#### x, psi functions ####
def relabel_x_to_psi(x, relabel_str):
    # may no longer work
    '''
    Relabels x according to relabel_str.

    Inputs:
        x: m x k np array
        relabel_str: string of format <old>-<new>

    Outputs:
        psi: bead labels np array

    Example:
    consider: <old> = AB, <new> = D, x is m x 3
    Any particle with both label A and label B, will be relabeled to have
    label D and neither A nor B. Label C will be unaffected.


    If len(<new>) = 1, then LETTERS.find(new) must be >= k
    (i.e label <new> cannot be present in x already)

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
        psi = np.delete(x, old_label, axis = 1)

        for i in new_labels:
            where = np.logical_and(x[:, i] == 0, x[:, old_label] == 1)
            psi[:, i] += where

    return psi

def writeSeq(seq, format='%.8e'):
    m, k = seq.shape
    for j in range(k):
        np.savetxt(f'pcf{j+1}.txt', seq[:, j], fmt = format)

def main(args_file=None, args_tmp=None):
    args = getArgs(args_file, args_tmp)
    print(args)

    with open(args.config_ifile, 'rb') as f:
        config = json.load(f)

    if args.m == -1:
        # need to infer m
        if osp.exists('x.npy'):
            x = np.load('x.npy')
            args.m, _ = x.shape
        elif osp.exists('S.npy'):
            S = np.load('S.npy')
            args.m, _ = e.shape
        elif args.sample_folder is not None:
            y_file = osp.join(args.sample_folder, 'y.npy')
            if osp.exists(y_file):
                args.m = len(np.load(y_file))

        if args.m == -1:
            raise Exception('Could not infer m')
        else:
            print(f'inferred m = {args.m}')

    # save nbeads
    config['nbeads'] = args.m

    # infer k
    if osp.exists('psi.npy'):
        psi = np.load('psi.npy')
        _, args.k = psi.shape
    elif osp.exists('x.npy'):
        x = np.load('x.npy')
        _, args.k = x.shape
    else:
        args.k = 0
    print(f'inferred k = {args.k}')

    sample_config = None # ground truth config file
    # used to copy TICG seed and diag chis
    if args.sample_folder is not None:
        sample_config_file = osp.join(args.sample_folder, 'config.json')
        if osp.exists(sample_config_file):
            with open(sample_config_file, 'rb') as f:
                sample_config = json.load(f)

        # copy over ground truth y
        y_gt_file = osp.join(args.sample_folder, 'y.npy')
        if osp.exists(y_gt_file):
            y_gt = crop(np.load(y_gt_file), args.m)
            print(f'y_gt.shape = {y_gt.shape}')
            np.save('y_gt.npy', y_gt)

    # load chi
    chi_file = 'chis.npy'
    if args.use_ground_truth_chi:
        assert args.sample_folder is not None
        args.chi = np.load(osp.join(args.sample_folder, 'chis.npy'))
        _, k = args.chi.shape
        np.save(chi_file, args.chi)
        assert k == args.k, f"cols of ground truth chi {args.k} doesn't match cols of seq {k}"
    elif osp.exists(chi_file):
        args.chi = np.load(chi_file)

        if args.max_ent:
            with open('chis.txt', 'w', newline='') as f:
                wr = csv.writer(f, delimiter = '\t')
                wr.writerow(args.chi[np.triu_indices(args.k)])
                wr.writerow(args.chi[np.triu_indices(args.k)])
    else:
        args.chi = None

    # save dense_diagonal
    config['dense_diagonal_on'] = args.dense_diagonal_on

    # load diag chi
    if osp.exists('diag_chis.npy'):
        config["diagonal_on"] = True
        if args.max_ent:
            diag_chis = np.load('diag_chis.npy')
            print(f'diag_chis loaded with shape {diag_chis.shape}')
            with open('chis_diag.txt', 'w', newline='') as f:
                wr = csv.writer(f, delimiter = '\t')
                wr.writerow(diag_chis)
                wr.writerow(diag_chis)
    else:
        config["diagonal_on"] = False

    if args.use_dmatrix:
        diag_chis = np.load('diag_chis.npy')
        if len(diag_chis) == args.m:
            D = calculate_D(diag_chis)
        else:
            diag_chis_step = calculate_diag_chi_step(config, diag_chis)
            D = calculate_D(diag_chis_step)

    # save diag_pseudobeads_on
    config['diag_pseudobeads_on'] = args.diag_pseudobeads_on

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

    if args.s_constant != 0 or args.e_constant != 0:
        raise Exception('deprecated')

    # set up e, s
    if psi is not None:
        writeSeq(psi)

        # save seq
        config['bead_type_files'] = [f'pcf{i}.txt' for i in range(1, args.k+1)]

        # save nspecies
        config["nspecies"] = args.k

        # save chi to config
        config['chis'] = args.chi.tolist()
    elif args.use_smatrix:
        config['bead_type_files'] = None
        config["nspecies"] = 0
    else:
        config['plaid_on'] = False
        config['bead_type_files'] = None
        config["nspecies"] = 0

    if args.use_dmatrix:
        config['dmatrix_on'] = True
    else:
        config['dmatrix_on'] = False
    if args.use_lmatrix:
        config['lmatrix_on'] = True
    else:
        config['lmatrix_on'] = False
    if args.use_smatrix:
        config['smatrix_on'] = True
    else:
        config['smatrix_on'] = False

    if not config['plaid_on'] and not config["diagonal_on"]:
        config['nonbonded_on'] = False

    # save bond type
    if args.bond_type is None:
        config['bond_type'] = 'none'
        config['bonded_on'] = False
        config["displacement_on"] = True
        config["translation_on"] = False
        config["crankshaft_on"] = False
        config["pivot_on"] = False
        config["rotate_on"] = False
    else:
        config['bond_type'] = args.bond_type
        if args.bond_type == 'gaussian':
            config["rotate_on"] = False

    # save nSweeps
    if args.n_sweeps is not None:
        config['nSweeps'] = args.n_sweeps

    # save phi_chromatin
    if args.phi_chromatin is None:
        assert args.volume is not None
        config['target_volume'] = args.volume
    else:
        config['phi_chromatin'] = args.phi_chromatin

    # save bond_length
    bond_length_file = 'bond_length.txt'
    if osp.exists(bond_length_file):
        args.bond_length = float(np.loadtxt(bond_length_file))
        print('Bond_length:', args.bond_length)
    config['bond_length'] = args.bond_length

    # save boundary_type
    config['boundary_type'] = args.boundary_type
    if args.aspect_ratio != 1:
        assert config['boundary_type'] == 'spheroid'
        config['aspect_ratio'] = args.aspect_ratio

    # save dump frequency
    if args.dump_frequency is not None:
        config['dump_frequency'] = args.dump_frequency
    if args.dump_stats_frequency is not None:
        config['dump_stats_frequency'] = args.dump_stats_frequency

    # save track_contactmap
    config['track_contactmap'] = args.track_contactmap

    # save gridmove_on
    config['gridmove_on'] = args.gridmove_on

    # save grid_size
    grid_size = None
    if osp.exists(args.grid_size):
        if args.grid_size.endswith('json'):
            with open(args.grid_size, 'r') as f:
                tmp = json.load(f)
                grid_size = tmp['grid_size']
        elif args.grid_size.endswith('txt'):
            arr = np.loadtxt(args.grid_size)
            arr = np.atleast_1d(arr)
            print(f'shape of args.grid_size = {arr.shape}')
            if len(arr) > 1:
                grid_size = arr[-1]
            else:
                grid_size = arr.item()
    elif args.grid_size.startswith('scale_'):
        scale = float(args.grid_size[6:])
        grid_size = args.bond_length * scale
    elif args.grid_size.lower() == 'none':
        assert args.sample_folder is not None
        if args.bond_length.is_integer():
            b = int(args.bond_length)
        else:
            b = args.bond_length
        if args.k_angle != 0:
            optimize_folder = osp.join(args.sample_folder,
                        f'optimize_grid_angle_{args.k_angle}_b_{b}_phi_{args.phi_chromatin}')
        else:
            optimize_folder = osp.join(args.sample_folder,
                        f'optimize_grid_b_{b}_phi_{args.phi_chromatin}')
        arr = np.loadtxt(osp.join(optimize_folder, 'grid_size.txt'))
        arr = np.atleast_1d(arr)
        grid_size = arr.item()
    else:
        grid_size = float(args.grid_size)
    if grid_size is None:
        raise Exception(f'Invalid grid_size {args.grid_size}')
    config['grid_size'] = grid_size

    # save k_angle
    if args.k_angle != 0:
        config['angles_on'] = True
        config['k_angle'] = args.k_angle

    # save bead volume
    config['beadvol'] = args.bead_vol

    # save update_contacts_distance
    config['update_contacts_distance'] = args.update_contacts_distance

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

    config['constant_chi'] = args.constant_chi
    if args.constant_chi > 0:
        config['constant_chi_on'] = True
    if args.max_ent and args.mode == 'all':
        config['constant_chi_on'] = True # turn on even if value = 0
        with open('chi_constant.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow([args.constant_chi])
            wr.writerow([args.constant_chi])
    if args.max_ent and args.mode.startswith('grid_size'):
        with open('grid_size.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow([args.grid_size])
            wr.writerow([args.grid_size])


    # save parallel
    config['parallel'] = args.parallel
    config['num_threads'] = args.num_threads


    with open(args.config_ofile, 'w') as f:
        json.dump(config, f, indent = 2)


#### test functions ####
class Tester():
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
        psi[:, 9] = np.logical_and(np.logical_and(x[:, 6], x[:, 7]),
                                    np.logical_not(x[:, 4])) # G and H and not E
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
            result = polynomial_kernel(x[i].reshape(-1, 1), x[i].reshape(-1, 1),
                                        degree=1, coef0=0)
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
    # Tester.test_nonlinear_chi()
    # Tester.test_polynomial_chi()
