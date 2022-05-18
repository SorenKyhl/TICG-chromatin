import argparse
import csv
import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import scipy.stats as ss
from seq2contact import (ArgparserConverter, load_E_S, load_final_max_ent_chi,
                         load_Y)
from sklearn.metrics import mean_squared_error

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
METHODS = ['ground_truth', 'random', 'PCA', 'PCA_split', 'kPCA', 'RPCA', 'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM']
SMALL_METHODS = {'ground_truth', 'random', 'PCA', 'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM'}
LABELS = ['Ground Truth', 'Random', 'PCA', 'PCA Split', 'kPCA', 'RPCA', 'K-means', 'NMF', 'GNN', 'Epigenetic', 'ChromHMM']

def getArgs(data_folder = None, sample = None, samples = None):
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    parser.add_argument('--data_folder', type=str, default=data_folder, help='location of input data')
    parser.add_argument('--sample', type=str, default=sample, help='sample id')
    parser.add_argument('--samples', type=AC.str2list, default=samples, help='list of sample ids separated by -')
    parser.add_argument('--sample_folder', type=str, help='location of input data')
    parser.add_argument('--ref_mode', type=str, help='deprecated')

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
    '''
    print('\nLOADING DATA')
    data = defaultdict(lambda: defaultdict(lambda : defaultdict(list))) # k, method, metric : list of replicate arrays

    for sample in args.samples:
        sample_folder = osp.join(args.data_folder, 'samples', f'sample{sample}')
        ground_truth_e, _ = load_E_S(sample_folder, throw_exception = False)
        ground_truth_y, _ = load_Y(sample_folder)
        for method in os.listdir(sample_folder):
            method_folder = osp.join(sample_folder, method)
            # methods should be formatted such that method.split('-')[0] is in METHODS
            if osp.isdir(method_folder) and method.split('-')[0] in METHODS:
                print(method)
                found_results = False # if not results found, don't add to dictionary
                for k_file in os.listdir(method_folder):
                    k_folder = osp.join(method_folder, k_file)
                    if osp.isdir(k_folder) and k_file.startswith('k'):
                        k = ArgparserConverter.str2int(k_file[1:])
                        if k is None:
                            k = 0
                        replicate_data = defaultdict(list)
                        for replicate in os.listdir(k_folder):
                            replicate_folder = osp.join(k_folder, replicate)
                            json_file = osp.join(replicate_folder, 'distance_pearson.json')
                            if osp.exists(json_file):
                                found_results = True
                                with open(json_file, 'r') as f:
                                    results = json.load(f)
                                    replicate_data['overall_pearson'].append(results['overall_pearson'])
                                    replicate_data['scc'].append(results['scc'])
                                    replicate_data['avg_dist_pearson'].append(results['avg_dist_pearson'])
                            else:
                                print(f"\tMISSING: {json_file}")
                                continue

                            rmse_e = None
                            if ground_truth_e is not None:
                                # load bead types
                                psi_file = osp.join(replicate_folder, 'resources', 'x.npy')
                                if osp.exists(psi_file):
                                    psi = np.load(psi_file)
                                else:
                                    psi = None
                                    print(f'\tpsi not found for {replicate_folder}')

                                # load chi
                                chi = load_final_max_ent_chi(k, replicate_folder, throw_exception = False)

                                # load energy
                                e, s = load_E_S(replicate_folder, psi = psi, chi = chi, throw_exception = False)

                                if e is not None:
                                    rmse_e = mean_squared_error(ground_truth_e, e, squared = False)
                            replicate_data['rmse-e'].append(rmse_e)

                            rmse_y = None
                            if ground_truth_y is not None:
                                y_file = osp.join(replicate_folder, 'y.npy')
                                if osp.exists(y_file):
                                    y = np.load(y_file)
                                    rmse_y = mean_squared_error(ground_truth_y, y, squared = False)

                                else:
                                    print(f'\ty not found for {replicate_folder}')
                            replicate_data['rmse-y'].append(rmse_y)

                            time = None
                            bash_file = osp.join(replicate_folder, 'bash.log')
                            if osp.exists(bash_file):
                                with open(bash_file, 'r') as f:
                                    for line in f:
                                        if 'finished entire simulation' in line:
                                            left = line.find('(')
                                            right = line.find(')')
                                            time = line[left+1:right].split(' ')[0]
                                            time = int(time)
                                            time /= 60 # minutes
                            replicate_data['time'].append(time)


                        # append replicate array to dictionary
                        if found_results:
                            data[k][method]['overall_pearson'].append(replicate_data['overall_pearson'])
                            data[k][method]['scc'].append(replicate_data['scc'])
                            data[k][method]['avg_dist_pearson'].append(replicate_data['avg_dist_pearson'])
                            data[k][method]['rmse-e'].append(replicate_data['rmse-e'])
                            data[k][method]['rmse-y'].append(replicate_data['rmse-y'])
                            data[k][method]['time'].append(replicate_data['time'])
    return data

def makeLatexTable(data, ofile, header = '', small = False, mode = 'w', sample_id = None):
    '''
    Writes data to ofile in latex table format.

    Inputs:
        data: dictionary containing data from loadData
        ofile: location to write data
        small: True to output smaller table with only methods in SMALL_METHODS and only SCC as metric')
        mode: file mode (e.g. 'w', 'a')
        sample_id: sample_id for table header; if set, table will show stdev over replicates for that sample
        ref: which method to use as referencem supported options: {"ground truth", "GNN"}
    '''
    print('\nMAKING TABLE')
    header = header.replace('_', "\_")
    with open(ofile, mode) as o:
        # set up first rows of table
        o.write("\\begin{center}\n")
        if small:
            metrics = ['scc', 'rmse-e', 'rmse-y', 'time']
            o.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
            o.write("\\hline\n")
            o.write("\\multicolumn{6}{|c|}{" + header + "} \\\ \n")
            if sample_id is not None:
                o.write("\\hline\n")
                o.write("\\multicolumn{6}{|c|}{sample " + f'{sample_id}' + "} \\\ \n")
            o.write("\\hline\n")
            o.write("Method & $\\ell$ & SCC & RMSE-E & RMSE-Y & TIME \\\ \n")
        else:
            metrics = ['overall_pearson', 'avg_dist_pearson', 'scc', 'rmse-e', 'rmse-y', 'time']
            o.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
            o.write("\\hline\n")
            o.write("\\multicolumn{8}{|c|}{" + header + "} \\\ \n")
            if sample_id is not None:
                o.write("\\hline\n")
                o.write("\\multicolumn{8}{|c|}{sample " + f'{sample_id}' + "} \\\ \n")
            o.write("\\hline\n")
            o.write("Method & k & Pearson R & Avg Dist Pearson R & SCC & RMSE-E & RMSE-Y & TIME\\\ \n")
        o.write("\\hline\\hline\n")

        # get reference data
        ground_truth_ref = None
        if 0 in data.keys() and 'ground_truth-S' in data[0]:
            ground_truth_ref = data[0]['ground_truth-S']
            print('ground truth found')
        elif 0 in data.keys() and 'ground_truth-E' in data[0]:
            ground_truth_ref = data[0]['ground_truth-E']
            print('ground truth found')
        else:
            # look for ground_truth-psi-chi
            for key in data.keys():
                if 'ground_truth-psi-chi' in data[key]:
                    ground_truth_ref = data[key]['ground_truth-psi-chi']
                    print('ground truth found')
                    break
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
                        assert sample_results.shape[0] == 1, f"label {label}, metric {metric}, k {k_label}, results {data[k][key][metric]}"
                        result = sample_results.reshape(-1)
                        if GNN_ref is not None:
                            ref_result = nested_list_to_array(GNN_ref[metric]).reshape(-1)
                    else:
                        try:
                            result = np.nanmean(sample_results, axis = 1)
                            if GNN_ref is not None:
                                ref_result = np.nanmean(nested_list_to_array(GNN_ref[metric]), axis = 1)
                                if len(result) > 1:
                                    stat, pval = ss.ttest_rel(ref_result, result)
                                    if pval < 0.05:
                                        significant = True
                        except Exception as e:
                            print(f'method {key}, k {k}, metric: {metric}')
                            print('GNN ref', GNN_ref[metric])
                            print('sample results', sample_results)
                            raise

                    # if GNN_ref is not None and metric == 'scc':
                    #     try:
                    #         delta_result = ref_result - result
                    #         delta_result_mean = np.round(np.nanmean(delta_result), 3)
                    #     except ValueError as e:
                    #         print(GNN_ref[metric])
                    #         print(f'method {key}, k {k}, metric: {metric}')
                    #         raise
                    #         delta_result_mean = None
                    # else:
                    #     delta_result_mean = None

                    result_mean = np.round(np.nanmean(result), 3)
                    if len(result) > 1:
                        result_std = np.round(np.nanstd(result), 3)
                        # if delta_result_mean is not None:
                        #     delta_result_std = np.round(np.nanstd(delta_result), 3)
                        # else:
                        #     delta_result_std = None
                        text += f" & {result_mean} $\pm$ {result_std}"
                        # if metric == 'scc':
                        #     text += f" & {delta_result_mean} $\pm$ {delta_result_std}"
                        if significant:
                            text += f' *{np.round(pval, 3)}'
                    else:
                        text += f" & {result_mean}"
                        # if metric == 'scc':
                        #     text += f" & {delta_result_mean}"

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
    print(args)

    fname = 'max_ent_table.txt'
    dataset = osp.split(args.data_folder)[1]

    if args.samples is not None:
        data = loadData(args)
        ofile = osp.join(args.data_folder, fname)

        makeLatexTable(data, ofile, dataset, small = True, mode = 'w')
        makeLatexTable(data, ofile, dataset, small = False, mode = 'a')

    if args.sample is not None:
        args.samples = [args.sample]
        data = loadData(args)
        ofile = osp.join(args.sample_folder, fname)

        makeLatexTable(data, ofile, dataset, small = True, mode = 'w', sample_id = args.sample)
        makeLatexTable(data, ofile, dataset, small = False, mode = 'a', sample_id = args.sample)



if __name__ == '__main__':
    main()
