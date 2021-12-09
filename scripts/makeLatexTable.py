import os
import os.path as osp
import sys

import json
import argparse
from collections import defaultdict
import numpy as np

from sklearn.metrics import mean_squared_error

# ensure that I can get_config
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from knightRuiz import knightRuiz
from get_config import str2bool, str2int, calculate_S

METHODS = ['ground_truth', 'random', 'PCA', 'PCA_split', 'kPCA', 'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM']
SMALL_METHODS = {'ground_truth', 'random', 'PCA', 'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM'}
LABELS = ['Ground Truth', 'Random', 'PCA', 'PCA Split', 'kPCA', 'K-means', 'NMF', 'GNN', 'Epigenetic', 'ChromHMM']


def getArgs(data_folder=None, sample=None, samples = None):
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--data_folder', type=str, default=data_folder, help='location of input data')
    parser.add_argument('--sample', type=int, default=sample, help='sample id')
    parser.add_argument('--samples', type=str2list, default=samples, help='list of sample ids separated by -')
    parser.add_argument('--sample_folder', type=str, help='location of input data')

    args = parser.parse_args()

    assert args.sample is not None or args.samples is not None, "either sample or samples must be set"
    if args.sample_folder is None and args.sample is not None:
        args.sample_folder = osp.join(args.data_folder, 'samples', f'sample{args.sample}')

    return args

def str2list(v, sep = '-'):
    """
    Helper function for argparser, converts str to list by splitting on sep.

    Exmaple for sep = '-': "i-j-k" -> [i,j,k]

    Inputs:
        v: string
        sep: separator
    """
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() == 'none':
            return None
        else:
            result = [i for i in v.split(sep)]
            for i, val in enumerate(result):
                if val.isnumeric():
                    result[i] = int(val)
            return result
    else:
        raise argparse.ArgumentTypeError('str value expected.')

def loadData(args):
    '''
    Loads data from args.data_folder for each sample <i> in args.samples.

    Data is expected to be found at args.data_folder/samples/sample<i>/<method>.

    <method> must be formatted such that <method>.split('-')[0] \in METHODS (e.g. 'nmf-binarize'.split('-')[0] = 'nmf')

    All directories within <method> that start with 'k' or 's' are assumed to contain replicate directories.
    All replicate directories are assumed to contain 'distance_pearson.json', from which data is loaded.
    '''
    data = defaultdict(lambda: defaultdict(lambda : defaultdict(list)))

    for sample in args.samples:
        sample_folder = osp.join(args.data_folder, 'samples', f'sample{sample}')
        for method in os.listdir(sample_folder):
            method_folder = osp.join(sample_folder, method)
            # methods should be formatted such that method.split('-')[0] is in METHODS
            if osp.isdir(method_folder) and method.split('-')[0] in METHODS:
                print(method)
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
                                print(f"Loading: {json_file}")
                                with open(json_file, 'r') as f:
                                    results = json.load(f)
                                    replicate_data['overall_pearson'].append(results['overall_pearson'])
                                    replicate_data['scc'].append(results['scc'])
                                    replicate_data['avg_dist_pearson'].append(results['avg_dist_pearson'])
                            else:
                                print(f"MISSING: {json_file}")

                            # look for s_matrix
                            s_matrix_file1 = osp.join(replicate_folder, 'resources', 's.npy')
                            s_matrix_file2 = osp.join(replicate_folder, 'resources', 's_matrix.txt')
                            if osp.exists(s_matrix_file1):
                                s = np.load(s_matrix_file1)
                            elif osp.exists(s_matrix_file2):
                                s = np.loadtxt(s_matrix_file2)
                            else:
                                # load bead types
                                x_file = osp.join(replicate_folder, 'resources', 'x.npy')
                                x = np.load(x_file)
                                _, k = x.shape

                                # load chi
                                chi_file = osp.join(replicate_folder, 'chis.txt')
                                allchis = np.atleast_2d(np.loadtxt(chi_file))
                                if len(allchis[0]) == 0:
                                    # shape will be wrong if k = 1
                                    allchis = allchis.T
                                lastchis = allchis[-1]
                                chi = np.zeros((k,k))
                                chi[np.triu_indices(k)] = lastchis # upper traingular chi

                                # caculate s
                                s = calculate_S(x, chi)

                            replicate_data['s'].append(s)

                        data[k][method]['overall_pearson'].append(np.mean(replicate_data['overall_pearson']))
                        data[k][method]['scc'].append(np.mean(replicate_data['scc']))
                        data[k][method]['avg_dist_pearson'].append(np.mean(replicate_data['avg_dist_pearson']))
                        data[k][method]['s'].append(replicate_data['s'])
                print('') # just making output look nicer
    return data

def makeLatexTable(data, ofile, header = '', small = False, mode = 'w'):
    '''
    Writes data to ofile in latex table format.

    Inputs:
        data: dictionary containing data from loadData
        ofile: location to write data
        small: True to output smaller table with only methods in SMALL_METHODS and only SCC as metric')
        mode: file mode (e.g. 'w', 'a')
        delta: True to compute difference in statistics relative to ground_truth-S
    '''
    header = header.replace('_', "\_")
    with open(ofile, mode) as o:
        # set up first rows of table
        o.write("\\begin{center}\n")
        if small:
            metrics = ['scc', 's']
            o.write("\\begin{tabular}{|c|c|c|c|c|}\n")
            o.write("\\hline\n")
            o.write("\\multicolumn{5}{|c|}{" + header + "} \\\ \n")
            o.write("\\hline\n")
            o.write("Method & k & SCC & $\\Delta$ SCC & MSE-S \\\ \n")
        else:
            metrics = ['overall_pearson', 'avg_dist_pearson', 'scc', 's']
            o.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
            o.write("\\hline\n")
            o.write("\\multicolumn{7}{|c|}{" + header + "} \\\ \n")
            o.write("\\hline\n")
            o.write("Method & k & Pearson R & Avg Dist Pearson R & SCC & $\\Delta$  SCC & MSE-S \\\ \n")
        o.write("\\hline\\hline\n")

        # get reference data
        if 'ground_truth-S' in data[0]:
            ref = data[0]['ground_truth-S']
            print('ref found')
        else:
            ref = None
            print('ref missing')

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
                    if metric == 'scc':
                        use_delta = True
                    else:
                        use_delta = False

                    if metric == 's':
                        if ref is None:
                            continue
                            # skip this if no ref
                        s_list_sample = data[k][key][metric]
                        ref_s_list_sample = ref[metric]
                        sample_results = []
                        for s_list, ref_s_list in zip(s_list_sample, ref_s_list_sample): # iterates over samples
                            replicate_results = []
                            for s, ref_s in zip(s_list, ref_s_list): # iterates over replicates
                                replicate_results.append(mean_squared_error(ref_s, s))
                            sample_results.append(np.mean(replicate_results))

                        result = np.array(sample_results)
                        result_mean = np.round(np.mean(result), 3)
                    else:
                        result = np.array(data[k][key][metric])
                        result_mean = np.round(np.mean(result), 3)

                        if ref is not None and use_delta:
                            try:
                                ref_result = np.array(ref[metric])
                                delta_result = ref_result - result
                                delta_result_mean = np.round(np.mean(delta_result), 3)
                            except ValueError as e:
                                print(e)
                                print(f'method {key}, k {k}, metric: {metric}')
                                delta_result_mean = None
                        else:
                            delta_result_mean = None

                    if len(result) > 1:
                        result_std = np.round(np.std(result), 3)
                        if delta_result_mean is not None:
                            delta_result_std = np.round(np.std(delta_result), 3)
                        else:
                            delta_result_std = None
                        text += f" & {result_mean} $\pm$ {result_std}"
                        if use_delta:
                            text += f" & {delta_result_mean} $\pm$ {delta_result_std}"
                    else:
                        text += f" & {result_mean}"
                        if use_delta:
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

def main():
    args = getArgs(data_folder='../sequences_to_contact_maps/dataset_10_27_21', samples=[40, 1230, 1718])
    print(args)

    fname = 'max_ent_table.txt'

    if args.samples is not None:
        data = loadData(args)
        ofile = osp.join(args.data_folder, fname)

    if args.sample is not None:
        args.samples = [args.sample]
        data = loadData(args)
        ofile = osp.join(args.sample_folder, fname)

    dataset = osp.split(args.data_folder)[1]

    mode = 'w'
    first = True
    for is_small in [True, False]:
        makeLatexTable(data, ofile, dataset, small = is_small, mode = mode)
        if first:
            mode = 'a'
            first = False


if __name__ == '__main__':
    main()
