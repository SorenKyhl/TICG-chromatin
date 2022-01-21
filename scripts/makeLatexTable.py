import os
import os.path as osp
import sys

import json
import argparse
from collections import defaultdict
import numpy as np
import scipy.stats as ss # ttest_rel

from sklearn.metrics import mean_squared_error

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.argparseSetup import str2int, str2bool, str2list
from neural_net_utils.utils import calculate_S

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
METHODS = ['ground_truth', 'random', 'PCA', 'PCA_split', 'kPCA', 'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM']
SMALL_METHODS = {'ground_truth', 'random', 'PCA', 'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM'}
LABELS = ['Ground Truth', 'Random', 'PCA', 'PCA Split', 'kPCA', 'K-means', 'NMF', 'GNN', 'Epigenetic', 'ChromHMM']

def getArgs(data_folder = None, sample = None, samples = None):
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--data_folder', type=str, default=data_folder, help='location of input data')
    parser.add_argument('--sample', type=int, default=sample, help='sample id')
    parser.add_argument('--samples', type=str2list, default=samples, help='list of sample ids separated by -')
    parser.add_argument('--sample_folder', type=str, help='location of input data')
    parser.add_argument('--ref_mode', type=str, help='deprecated')

    args = parser.parse_args()

    assert args.sample is not None or args.samples is not None, "either sample or samples must be set"
    if args.sample_folder is None and args.sample is not None:
        args.sample_folder = osp.join(args.data_folder, 'samples', f'sample{args.sample}')

    return args

def load_chi(replicate_folder, k):
    # find final it
    max_it = -1
    for file in os.listdir(replicate_folder):
        if osp.isdir(osp.join(replicate_folder, file)) and file.startswith('iteration'):
            it = int(file[9:])
            if it > max_it:
                max_it = it

    if max_it < 0:
        raise Exception(f'max it not found for {replicate_folder}')

    config_file = osp.join(replicate_folder, f'iteration{max_it}', 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'rb') as f:
            config = json.load(f)
    else:
        return None

    chi = np.zeros((k,k))
    for i, bead_i in enumerate(LETTERS[:k]):
        for j in range(i,k):
            bead_j = LETTERS[j]
            chi[i,j] = config[f'chi{bead_i}{bead_j}']

    return chi

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
        for method in os.listdir(sample_folder):
            method_folder = osp.join(sample_folder, method)
            # methods should be formatted such that method.split('-')[0] is in METHODS
            if osp.isdir(method_folder) and method.split('-')[0] in METHODS:
                print(method)
                found_results = False # if not results found, don't add to dictionary
                for k_file in os.listdir(method_folder):
                    k_folder = osp.join(method_folder, k_file)
                    if osp.isdir(k_folder) and k_file.startswith('k'):
                        k = str2int(k_file[1:])
                        if k is None:
                            k = 0
                        replicate_data = defaultdict(list)
                        for replicate in os.listdir(k_folder):
                            replicate_folder = osp.join(k_folder, replicate)
                            json_file = osp.join(replicate_folder, 'distance_pearson.json')
                            if osp.exists(json_file):
                                found_results = True
                                # print(f"Loading: {json_file}")
                                with open(json_file, 'r') as f:
                                    results = json.load(f)
                                    replicate_data['overall_pearson'].append(results['overall_pearson'])
                                    replicate_data['scc'].append(results['scc'])
                                    replicate_data['avg_dist_pearson'].append(results['avg_dist_pearson'])
                            else:
                                print(f"\tMISSING: {json_file}")
                                continue

                            # look for s_matrix
                            s_matrix_file1 = osp.join(replicate_folder, 'resources', 's.npy')
                            s_matrix_file2 = osp.join(replicate_folder, 'resources', 's_matrix.txt')
                            if osp.exists(s_matrix_file1):
                                s = np.load(s_matrix_file1)
                            elif osp.exists(s_matrix_file2):
                                s = np.loadtxt(s_matrix_file2)
                            else:
                                # load bead types
                                # x_file1 = osp.join(replicate_folder, 'resources', 'x_linear.npy')
                                x_file2 = osp.join(replicate_folder, 'resources', 'x.npy')
                                # if osp.exists(x_file1):
                                #     x = np.load(x_file1)
                                if osp.exists(x_file2):
                                    x = np.load(x_file2)
                                else:
                                    print(f'\tx not found for {replicate_folder}')
                                    continue

                                # load chi
                                chi = load_chi(replicate_folder, k)

                                # caculate s
                                s = calculate_S(x, chi)

                            replicate_data['s'].append(s)

                        # append replicate array to dictionary
                        if found_results:
                            data[k][method]['overall_pearson'].append(replicate_data['overall_pearson'])
                            data[k][method]['scc'].append(replicate_data['scc'])
                            data[k][method]['avg_dist_pearson'].append(replicate_data['avg_dist_pearson'])
                            data[k][method]['s'].append(replicate_data['s'])
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
            metrics = ['scc', 's']
            o.write("\\begin{tabular}{|c|c|c|c|c|}\n")
            o.write("\\hline\n")
            o.write("\\multicolumn{5}{|c|}{" + header + "} \\\ \n")
            if sample_id is not None:
                o.write("\\hline\n")
                o.write("\\multicolumn{5}{|c|}{sample " + f'{sample_id}' + "} \\\ \n")
            o.write("\\hline\n")
            o.write("Method & $\\ell$ & SCC & $\\Delta$ SCC & MSE-S \\\ \n")
        else:
            metrics = ['overall_pearson', 'avg_dist_pearson', 'scc', 's']
            o.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
            o.write("\\hline\n")
            o.write("\\multicolumn{7}{|c|}{" + header + "} \\\ \n")
            if sample_id is not None:
                o.write("\\hline\n")
                o.write("\\multicolumn{7}{|c|}{sample " + f'{sample_id}' + "} \\\ \n")
            o.write("\\hline\n")
            o.write("Method & k & Pearson R & Avg Dist Pearson R & SCC & $\\Delta$  SCC & MSE-S \\\ \n")
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
                    # TODO get best GNN
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
                    if metric == 's':
                        if ground_truth_ref is None:
                            continue
                            # skip this if no ref
                        s_list_sample = data[k][key][metric] # list of length samples, each entry is a list of replicates
                        ref_s_list_sample = ground_truth_ref[metric]
                        sample_results = []
                        for s_list, ref_s_list in zip(s_list_sample, ref_s_list_sample): # iterates over samples
                            replicate_results = []
                            p_ref_s = -1
                            ref_s = ref_s_list[0] # all the same so just grab index 0
                            for s in s_list: # iterates over replicates
                                replicate_results.append(mean_squared_error(ref_s, s))
                            sample_results.append(replicate_results)

                        sample_results = nested_list_to_array(sample_results)
                    else:
                        sample_results = nested_list_to_array(data[k][key][metric])

                    if sample_id is not None:
                        assert sample_results.shape[0] == 1, f"label {label}, metric {metric}, k {k_label}, results {sample_results}"
                        result = sample_results.reshape(-1)
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
                            print(sample_results)
                            raise

                    if GNN_ref is not None and metric == 'scc':
                        try:
                            delta_result = ref_result - result
                            delta_result_mean = np.round(np.nanmean(delta_result), 3)
                        except ValueError as e:
                            print(GNN_ref[metric])
                            print(f'method {key}, k {k}, metric: {metric}')
                            raise
                            delta_result_mean = None
                    else:
                        delta_result_mean = None

                    result_mean = np.round(np.nanmean(result), 3)
                    if len(result) > 1:
                        result_std = np.round(np.nanstd(result), 3)
                        if delta_result_mean is not None:
                            delta_result_std = np.round(np.nanstd(delta_result), 3)
                        else:
                            delta_result_std = None
                        text += f" & {result_mean} $\pm$ {result_std}"
                        if metric == 'scc':
                            text += f" & {delta_result_mean} $\pm$ {delta_result_std}"
                        if significant:
                            text += f' *{np.round(pval, 3)}'
                    else:
                        text += f" & {result_mean}"
                        if metric == 'scc':
                            text += f" & {delta_result_mean}"

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
