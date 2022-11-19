import argparse
import csv
import json
import os
import os.path as osp
import tarfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from compare_contact import plotDistanceStratifiedPearsonCorrelation
from scipy.stats import pearsonr
from seq2contact import (SCC, ArgparserConverter, DiagonalPreprocessing,
                         calc_dist_strat_corr, calculate_D,
                         calculate_diag_chi_step, calculate_E_S,
                         calculate_net_energy, crop, load_E_S,
                         load_max_ent_chi, load_Y, s_to_E)
from sklearn.metrics import mean_squared_error

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
METHODS = ['ground_truth', 'random', 'PCA', 'PCA_split', 'kPCA', 'RPCA',
            'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM']
SMALL_METHODS = {'ground_truth', 'random', 'PCA', 'k_means', 'nmf', 'GNN',
            'epigenetic', 'ChromHMM'}
LABELS = ['Ground Truth', 'Random', 'PCA', 'PCA Split', 'kPCA', 'RPCA',
            'K-means', 'NMF', 'GNN', 'Epigenetic', 'ChromHMM']

def getArgs(data_folder = None, sample = None, samples = None):
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    parser.add_argument('--data_folder', type=str, default=data_folder,
                            help='location of input data')
    parser.add_argument('--sample', type=str, default=sample,
                            help='sample id')
    parser.add_argument('--samples', type=AC.str2list, default=samples,
                            help='list of sample ids separated by -')
    parser.add_argument('--sample_folder', type=str,
                        help='location of input data')
    parser.add_argument('--replicate', type=AC.str2int,
                        help='which replicate to consider (None for all)')
    parser.add_argument('--experimental', action='store_true',
                        help="True for experimental data mode - ground truth won't be present")
    parser.add_argument('--convergence_definition', type=str,
                        help='key to define convergence {strict, normal, param} (None for all 3)')

    args = parser.parse_args()

    assert args.sample is not None or args.samples is not None, "either sample or samples must be set"
    if args.sample_folder is None and args.sample is not None:
        args.sample_folder = osp.join(args.data_folder, 'samples', f'sample{args.sample}')

    return args

def nested_list_to_array(nested_list):
    '''
    Safely converts nested list to np array when sublists are not all of same length (appends None)
    '''

    # find max length (i.e. max # of replicates for method)
    max_len = -1
    for sublist in nested_list:
        if len(sublist) > max_len:
            max_len = len(sublist)

    # append nan to short lists
    for sublist in nested_list:
        len_sublist = len(sublist)
        while len_sublist < max_len:
            sublist.append(None)
            len_sublist = len(sublist)

    return np.array(nested_list, dtype=float)

def loadData(args):
    '''
    Loads data from args.data_folder for each sample <i> in args.samples.

    Data is expected to be found at args.data_folder/samples/sample<i>/<method>.

    <method> must be formatted such that <method>.split('-')[0] \in METHODS (e.g. 'nmf-binarize'.split('-')[0] = 'nmf')

    All directories within <method> that start with 'k' or 's' are assumed to contain replicate directories.
    All replicate directories are assumed to contain 'distance_pearson.json', from which data is loaded.

    Inputs:
        args: parsed command line arguments
    Outputs:
        data: nested dictionary k, method, metric : list of replicate arrays
        converged_mask: mask for list of replicate arrays, 1 if ALL max ent runs converged
    '''
    print(f'\nLOADING DATA with {args.convergence_definition} convergence')
    data = defaultdict(lambda: defaultdict(lambda : defaultdict(list))) # k, method, metric : list of replicate arrays
    converged_mask = np.ones(len(args.samples)).astype(bool)

    for i, sample in enumerate(args.samples):
        sample_folder = osp.join(args.data_folder, 'samples', f'sample{sample}')
        diag_chi_continuous = np.load(osp.join(sample_folder, 'diag_chis_continuous.npy'))
        D = calculate_D(diag_chi_continuous)
        S = np.load(osp.join(sample_folder, 's.npy'))
        ground_truth_ED = calculate_net_energy(S, D)
        ground_truth_y, ground_truth_ydiag = load_Y(sample_folder)
        ground_truth_y = ground_truth_y.astype(np.float64) / np.max(np.diagonal(ground_truth_y))
        for method in os.listdir(sample_folder):
            method_folder = osp.join(sample_folder, method)
            # methods should be formatted such that method.split('-')[0] is in METHODS
            if osp.isdir(method_folder) and method.split('-')[0] in METHODS:
                print(method)
                for k_file in os.listdir(method_folder):
                    k_folder = osp.join(method_folder, k_file)
                    if osp.isdir(k_folder) and k_file.startswith('k'):
                        k = ArgparserConverter.str2int(k_file[1:])
                        if k is None:
                            k = 0
                        replicate_data = defaultdict(list)
                        for replicate in os.listdir(k_folder):
                            if args.replicate is not None and int(replicate[-1]) != args.replicate:
                                continue
                            print('\t', replicate)
                            replicate_folder = osp.join(k_folder, replicate)

                            # convergence
                            converged_it = None
                            if not osp.exists(osp.join(replicate_folder, 'iteration2')):
                                # if there was only one iteration use that as converged iteration
                                converged_path = osp.join(replicate_folder, f'iteration1')
                            else:
                                if args.convergence_definition == 'param':
                                    # param convergence
                                    chis_diag_file = osp.join(replicate_folder, 'chis_diag.txt')
                                    if osp.exists(chis_diag_file):
                                        diag_chis = np.loadtxt(chis_diag_file)
                                    else:
                                        diag_chis = None

                                    chis_file = osp.join(replicate_folder, 'chis.txt')
                                    if osp.exists(chis_file):
                                        chis = np.loadtxt(chis_file)
                                        params = np.concatenate((diag_chis, chis), axis = 1)
                                    else:
                                        params = diag_chis

                                    eps = 1
                                    for j in range(1, len(params)):
                                        diff = params[j] - params[j-1]
                                        if np.linalg.norm(diff, ord = 2) < eps:
                                            converged_it = j
                                            break
                                else:
                                    # loss convergence
                                    if args.convergence_definition == 'strict':
                                        eps = 1e-3
                                    elif args.convergence_definition == 'normal':
                                        eps = 1e-2
                                    else:
                                        raise Exception(f'Unrecognized convergence_definition: {args.convergence_definition}')

                                    convergence_file = osp.join(replicate_folder, 'convergence.txt')
                                    if osp.exists(convergence_file):
                                        conv = np.loadtxt(convergence_file)
                                        for j in range(1, len(conv)):
                                            diff = conv[j] - conv[j-1]
                                            if np.abs(diff) < eps and conv[j] < conv[0]:
                                                converged_it = j
                                                break

                                if converged_it is not None:
                                    converged_path = osp.join(replicate_folder, f'iteration{converged_it+1}') # converged path is iteration after max ent converged
                                else:
                                    converged_mask[i] = 0
                                    converged_path = None

                            # SCC
                            yhat = None
                            if converged_path is not None:
                                data_file = osp.join(converged_path, 'production_out.tar.gz')
                                if osp.exists(data_file):
                                    with tarfile.open(data_file) as f:
                                        f.extractall(converged_path)
                                y_file = osp.join(converged_path, 'production_out/contacts.txt')
                                if osp.exists(y_file):
                                    yhat = np.loadtxt(y_file)
                                    meanDist = DiagonalPreprocessing.genomic_distance_statistics(yhat)
                                    yhat_diag = DiagonalPreprocessing.process(yhat, meanDist, verbose = False)

                                    overall_corr, corr_scc, avg_diag = plotDistanceStratifiedPearsonCorrelation(ground_truth_y,
                                                                            yhat, ground_truth_ydiag, yhat_diag, converged_path)
                                    replicate_data['overall_pearson'].append(overall_corr)
                                    replicate_data['scc'].append(corr_scc)
                                    replicate_data['avg_dist_pearson'].append(avg_diag)
                                else:
                                    print(f"Didn't find {y_file}")

                            rmse_ED = None
                            if ground_truth_ED is not None:
                                if converged_it is not None:
                                    # load bead types
                                    psi_file = osp.join(replicate_folder, 'resources', 'x.npy')
                                    if osp.exists(psi_file):
                                        psi = np.load(psi_file)
                                    else:
                                        psi = None
                                        print(f'\tpsi not found for {replicate_folder}')

                                    # load chi
                                    chi = load_max_ent_chi(k, converged_path, throw_exception = False)

                                    # calc s
                                    _, S = calculate_E_S(psi, chi)

                                    with open(osp.join(converged_path, 'config.json'), 'r') as f:
                                        config = json.load(f)
                                    diag_chis_continuous = calculate_diag_chi_step(config)
                                    D = calculate_D(diag_chis_continuous)

                                    # calc s+d
                                    ED = calculate_net_energy(S, D)
                                elif converged_path is not None:
                                    # but converged_it is None
                                    # means this is a GNN/MLP result
                                    ematrix_file = osp.join(replicate_folder, 'resources/e_matrix.txt')
                                    smatrix_file = osp.join(replicate_folder, 'resources/s_matrix.txt')
                                    if osp.exists(ematrix_file):
                                        ED = np.loadtxt(ematrix_file)
                                    elif osp.exists(smatrix_file):
                                        SD = np.loadtxt(smatrix_file)
                                        ED = s_to_E(SD)
                                    else:
                                        raise Exception(f'no <e,s>_matrix.txt for {replicate_folder}')
                                else:
                                    ED = None

                                if ED is not None:
                                    rmse_ED = mean_squared_error(ground_truth_ED, ED, squared = False)

                            replicate_data['rmse-E+D'].append(rmse_ED)

                            rmse_y = None
                            if yhat is not None:
                                # normalize yhat
                                yhat = yhat.astype(np.float64) / np.mean(np.diagonal(yhat))
                                rmse_y = mean_squared_error(ground_truth_y, yhat, squared = False)
                            replicate_data['rmse-y'].append(rmse_y)

                            bash_file = osp.join(replicate_folder, 'bash.log')
                            if osp.exists(bash_file):
                                with open(bash_file, 'r') as f:
                                    times = []
                                    for line in f:
                                        if 'finished iteration' in line:
                                            it = int(line.split(' ')[2].replace(':', ''))
                                            left = line.find('(')
                                            right = line.find(')')
                                            t = int(line[left+1:right].split(' ')[0])
                                            times.append(t)

                            if converged_it is None:
                                converge_time = None
                                if converged_path is None:
                                    final_t = None
                                else:
                                    final_t = times[-1]/60
                            else:
                                converge_time = np.sum(times[:converged_it])
                                converge_time /= 60 # to minutes
                                final_t = times[converged_it] / 60
                            replicate_data['final_time'].append(final_t)
                            replicate_data['converged_time'].append(converge_time)
                            if converged_path is None:
                                replicate_data['total_time'].append(None)
                            elif converge_time is None:
                                replicate_data['total_time'].append(final_t)
                            else:
                                replicate_data['total_time'].append(converge_time + final_t)
                            replicate_data['converged_it'].append(converged_it)

                        # append replicate array to dictionary
                        data[k][method]['overall_pearson'].append(replicate_data['overall_pearson'])
                        data[k][method]['scc'].append(replicate_data['scc'])
                        data[k][method]['avg_dist_pearson'].append(replicate_data['avg_dist_pearson'])
                        data[k][method]['rmse-E+D'].append(replicate_data['rmse-E+D'])
                        data[k][method]['rmse-y'].append(replicate_data['rmse-y'])
                        data[k][method]['converged_time'].append(replicate_data['converged_time'])
                        data[k][method]['final_time'].append(replicate_data['final_time'])
                        data[k][method]['total_time'].append(replicate_data['total_time'])
                        data[k][method]['converged_it'].append(replicate_data['converged_it'])

    print(f'{np.sum(converged_mask)} out of {len(converged_mask)} converged')
    print(converged_mask)
    return data, converged_mask

def makeLatexTable(data, ofile, header, small, mode = 'w', sample_id = None,
                    experimental = False, nan_mask = None):
    '''
    Writes data to ofile in latex table format.

    Inputs:
        data: dictionary containing data from loadData
        ofile: location to write data
        small: True to output smaller table with only methods in SMALL_METHODS and only SCC as metric')
        mode: file mode (e.g. 'w', 'a')
        sample_id: sample_id for table header; if set, table will show stdev over replicates for that sample
        experimental: True if ground_truth won't be present
        nna_mask: mask for sample_results, True to set to nan
    '''
    print(f'\nMAKING TABLE-small={small}')
    header = header.replace('_', "\_")
    with open(ofile, mode) as o:
        # set up first rows of table
        o.write("\\begin{center}\n")
        if small:
            metrics = ['scc', 'rmse-E+D', 'rmse-y', 'total_time']
        else:
            metrics = ['scc', 'rmse-E+D', 'rmse-y',
                        'converged_it', 'converged_time', 'final_time']
        num_cols = len(metrics) + 2
        num_cols_str = str(num_cols)

        o.write("\\begin{tabular}{|" + "c|"*num_cols + "}\n")
        o.write("\\hline\n")
        o.write("\\multicolumn{" + num_cols_str + "}{|c|}{" + header + "} \\\ \n")
        if sample_id is not None:
            o.write("\\hline\n")
            o.write("\\multicolumn{" + num_cols_str + "}{|c|}{sample " + f'{sample_id}' + "} \\\ \n")

        o.write("\\hline\n")

        if small:
            o.write("Method & k & SCC & RMSE-Energy & RMSE-Y & Total Time \\\ \n")
        else:
            o.write("Method & k & SCC & RMSE-Energy & RMSE-Y & Converged It & Converged Time & Final Time \\\ \n")
        o.write("\\hline\\hline\n")

        # get reference data
        ground_truth_ref = None
        if not experimental:
            if 0 in data.keys() and 'ground_truth-diag_chi-E' in data[0]:
                ground_truth_ref = data[0]['ground_truth-diag_chi-E']
                print('ground truth found')
            else:
                print('ground truth missing')
                for key in data.keys():
                    print(f'key 1: {key}, key 2: {data[key].keys()}')

        GNN_ref = None
        if 0 in data.keys():
            for method in data[0].keys():
                if 'GNN' in method:
                    print(f'GNN found: using {method}')
                    GNN_ref = data[0][method]
                    # TODO get best GNN not first
                    break

        for k in sorted(data.keys()):
            first = True # only write k for first row in section
            keys, labels = sort_method_keys(data[k].keys())
            for key, label in zip(keys, labels):
                if small and key.split('-')[0] not in SMALL_METHODS:
                    # skip methods not in SMALL_METHODS
                    continue
                if nan_mask is not None and len(nested_list_to_array(data[k][key]['scc'])) != len(nan_mask):
                    # skip method is not performed for all samples
                    continue
                if first: # only write k for first row in section
                    k_label = k
                    if k_label == 0:
                        k_label = '-'
                    first = False
                else:
                    k_label = ''
                text = f"{label} & {k_label}"

                for metric in metrics:
                    significant = False # two sided t test
                    sample_results = nested_list_to_array(data[k][key][metric])

                    if sample_id is not None:
                        assert sample_results.shape[0] == 1, f"label {label}, metric {metric}, k {k_label}, results {data[k][key][metric]}, {sample_results}"
                        result = sample_results.reshape(-1)
                        if GNN_ref is not None:
                            ref_result = nested_list_to_array(GNN_ref[metric]).reshape(-1)
                    else:
                        if nan_mask is not None:
                            sample_results[nan_mask] = np.nan
                        try:
                            result = np.nanmean(sample_results, axis = 1)
                            if GNN_ref is not None:
                                ref_result = np.nanmean(nested_list_to_array(GNN_ref[metric]), axis = 1)
                                if nan_mask is not None:
                                    ref_result[nan_mask] = np.nan
                                if len(result) > 1:
                                    stat, pval = ss.ttest_rel(ref_result, result)
                                    if pval < 0.05:
                                        significant = True
                        except Exception as e:
                            print(f'method {key}, k {k}, metric: {metric}')
                            print('GNN ref', GNN_ref[metric])
                            print('sample results', sample_results)
                            raise

                    if 'time' in metric:
                        roundoff = 1
                    elif metric == 'rmse-E+D':
                        roundoff = 2
                    elif metric == 'rmse-y':
                        roundoff = 4
                    else:
                        roundoff = 3

                    result_mean = np.round(np.nanmean(result), roundoff)
                    if len(result) > 1:
                        result_std = np.round(np.nanstd(result), roundoff)
                        text += f" & {result_mean} $\pm$ {result_std}"
                        if significant:
                            text += f' *{np.round(pval, roundoff)}'
                    else:
                        text += f" & {result_mean}"

                text += " \\\ \n"

                o.write(text)
            o.write("\\hline\n")
        o.write("\\end{tabular}\n")
        o.write("\\end{center}\n\n")

def sort_method_keys(keys):
    '''Sorts keys to match order of METHODS and gets corresponding labels from LABELS.'''
    sorted_keys = []
    sorted_labels = []
    for method, label in zip(METHODS, LABELS):
        for key in keys:
            split = key.split('-')
            if split[0] == method:
                sorted_keys.append(key)
                if len(split) > 1:
                    sorted_labels.append(label + '-' + '-'.join(split[1:]))
                else:
                    sorted_labels.append(label)

    return sorted_keys, sorted_labels

def main(data_folder=None, sample=None):
    args = getArgs(data_folder = data_folder, sample = sample)
    assert args.sample is None or args.samples is None
    print('-'*100, '\n', args)

    if args.replicate is None:
        fname = 'max_ent_table.txt'
    else:
        fname = f'max_ent_table_{args.replicate}.txt'
    dataset = osp.split(args.data_folder)[1]

    if args.replicate is None:
        label = dataset
    else:
        label = dataset + f' replicate {args.replicate}'

    if args.sample is not None:
        args.samples = [args.sample]
        ofile = osp.join(args.sample_folder, fname)
    else:
        ofile = osp.join(args.data_folder, fname)

    if args.convergence_definition is None:
        convergence_def_list = ['normal', 'strict', 'param']
    else:
        convergence_def_list = [args.convergence_definition]

    mode = 'w'
    for defn in convergence_def_list:
        args.convergence_definition = defn
        temp_label = label + f' {defn} convergence'
        data, converged_mask = loadData(args)
        not_converged_mask = np.logical_not(converged_mask) # now 1 if not converged

        # small
        makeLatexTable(data, ofile, temp_label, True, mode,
                        sample_id = args.sample, experimental = args.experimental,
                        nan_mask = not_converged_mask)
        mode = 'a' # switch to append for remaning tables

        makeLatexTable(data, ofile, temp_label, False, mode,
                        sample_id = args.sample, experimental = args.experimental,
                        nan_mask = not_converged_mask)

if __name__ == '__main__':
    main()
