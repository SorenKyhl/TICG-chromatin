import os
import os.path as osp
import sys

import json
import argparse
from collections import defaultdict
import numpy as np

# ensure that I can get_config
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from knightRuiz import knightRuiz
from get_config import str2list, str2bool

METHODS = ['ground_truth', 'random', 'PCA', 'PCA_split', 'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM']
SMALL_METHODS = ['ground_truth', 'random', 'PCA', 'k_means', 'nmf', 'GNN', 'epigenetic', 'ChromHMM']
LABELS = ['Ground Truth', 'Random', 'PCA', 'PCA Split', 'K-means', 'NMF', 'GNN', 'Epigenetic', 'ChromHMM']


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--data_folder', type=str, default='../sequences_to_contact_maps/dataset_09_21_21', help='location of input data')
    parser.add_argument('--sample', type=int, default=1, help='sample id')
    parser.add_argument('--samples', type=str2list, help='list of sample ids separated by -')
    parser.add_argument('--sample_folder', type=str, help='location of input data')
    parser.add_argumet('--small', type=str2bool, default=False, help='True to output smaller table with only methods in SMALL_METHODS and SCC as metric')

    args = parser.parse_args()

    assert args.sample is not None or args.samples is not None, "either sample or samples must be set"
    if args.sample_folder is None and args.sample is not None:
        args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))

    return args

def loadData(args):
    '''
    Loads data from args.data_folder for each sample <i> in args.samples.

    Data is expected to be found at args.data_folder/samples/sample<i>/<method>.

    <method> must be formatted such that <method>.split('-')[0] \in METHODS (e.g. 'nmf-binarize'.split('-')[0] = 'nmf')

    All directories within <method> that start with 'k' are assumed to contain 'distance_pearson.json', from which data is loaded.
    '''
    data = defaultdict(lambda: defaultdict(lambda : defaultdict(list)))

    for sample in args.samples:
        sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(sample))
        for method in os.listdir(sample_folder):
            method_folder = osp.join(sample_folder, method)
            # methods should be formatted such that method.split('-')[0] is in METHODS
            if osp.isdir(method_folder) and method.split('-')[0] in METHODS:
                print(method)
                for k_file in os.listdir(method_folder):
                    k_folder = osp.join(method_folder, k_file)
                    if osp.isdir(k_folder) and k_file.startswith('k'):
                        k = int(k_file[1])
                        json_file = osp.join(k_folder, 'distance_pearson.json')
                        if osp.exists(json_file):
                            found_anything = True
                            print("Loading: {}".format(json_file))
                            with open(json_file, 'r') as f:
                                results = json.load(f)
                                data[k][method]['overall_pearson'].append(results['overall_pearson'])
                                data[k][method]['scc'].append(results['scc'])
                                data[k][method]['avg_dist_pearson'].append(results['avg_dist_pearson'])
                        else:
                            print("MISSING: {}".format(json_file))
                print('')
                # just making output look nicer
    return data

def makeLatexTable(data, ofile, small):
    '''Writes data to ofile in latex table format.'''
    with open(ofile, 'w') as o:
        o.write("\\begin{center}\n")
        if small:
            metrics = ['scc']
            o.write("\\begin{tabular}{|c|c|c|}\n")
            o.write("\\hline\n")
            o.write("Method & k & SCC \\\ \n")
        else:
            metrics = ['overall_pearson', 'avg_dist_pearson', 'scc']
            o.write("\\begin{tabular}{|c|c|c|c|c|}\n")
            o.write("\\hline\n")
            o.write("Method & k & Pearson R & Avg Dist Pearson R & SCC \\\ \n")
        o.write("\\hline\\hline\n")
        for k in sorted(data.keys()):
            first = True # only write k for first row in section
            keys, labels = sort_method_keys(data[k].keys())
            for key, label in zip(keys, labels):
                if small and key not in SMALL_METHODS:
                    continue
                if first:
                    k_label = k
                    first = False
                else:
                    k_label = ''
                text = "{} & {}".format(label, k_label)

                for metric in metrics:
                    data_list = avg_dist_pearson_list = data[k][key][metric]
                    data_mean = np.round(np.mean(data_list), 3)
                    if len(data_list) > 1:
                        data_std = np.round(np.std(data_list), 3)
                        text += " & {} $\pm$ {}".format(data_mean, data_std)
                    else:
                        text += " & {}".format(data_mean)

                text += " \\\ \n"

                o.write(text)
            o.write("\\hline\n")
        o.write("\\end{tabular}\n")
        o.write("\\end{center}")

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
                    sorted_labels.append(label + '-' + split[1])
                else:
                    sorted_labels.append(label)

    return sorted_keys, sorted_labels

def main():
    args = getArgs()

    if small:
        fname = 'max_ent_table_small.txt'
    else:
        fname = 'max_ent_table.txt'

    if args.samples is not None:
        data = loadData(args)
        ofile = osp.join(args.data_folder, fname)

    if args.sample is not None:
        args.samples = [args.sample]
        data = loadData(args)
        ofile = osp.join(args.sample_folder, fname)

    makeLatexTable(data, ofile, args.small)


if __name__ == '__main__':
    main()
