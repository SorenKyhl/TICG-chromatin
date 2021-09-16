import os
import os.path as osp

import json
import argparse

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--data_folder', type=str, default='../sequences_to_contact_maps/dataset_08_26_21', help='location of input data')
    parser.add_argument('--sample', type=int, help='sample id')
    parser.add_argument('--samples', type=str2list, help='list of sample ids separated by -')
    parser.add_argument('--sample_folder', type=str, help='location of input data')

    args = parser.parse_args()

    assert args.sample is not None or args.samples is not None, "either sample or samples must be set"
    if args.sample_folder is None and args.sample is not None:
        args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))

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


def makeLatexTableSample(args):
    methods = ['ground_truth', 'random', 'PCA', 'PCA_split', 'k_means', 'nmf', 'GNN']
    labels = ['Ground Truth', 'Random', 'PCA', 'PCA Split', 'K-means', 'NMF', 'GNN']

    ofile = osp.join(args.sample_folder, 'max_ent_table.txt')
    with open(ofile, 'w') as o:
        o.write("\\begin{center}\n")
        o.write("\\begin{tabular}{|c|c|c|c|c|}\n")
        o.write("\\hline\n")
        o.write("Method & k & Pearson R & Avg Dist Pearson R & SCC \\\ \n")
        o.write("\\hline\\hline\n")
        for k in range(1, 10):
            empty = True # change to False if any methods are found for given k
            first = True # only write k for first row in section
            for method, label in zip(methods, labels):
                json_file = osp.join(args.sample_folder, method, 'k{}'.format(k), 'distance_pearson.json')
                if osp.exists(json_file):
                    empty = False
                    with open(json_file, 'r') as f:
                        results = json.load(f)
                        overall_pearson = results['overall_pearson']
                        scc = results['scc']
                        avg_dist_pearson = results['avg_dist_pearson']

                    if first:
                        k_label = k
                        first = False
                    else:
                        k_label = ''
                    o.write("{} & {} & {} &  {} & {} \\\ \n".format(label, k_label, overall_pearson, avg_dist_pearson, scc))
            if not empty:
                o.write("\\hline\n")
        o.write("\\end{tabular}\n")
        o.write("\\end{center}")

def makeLatexTableSamples(args):
    methods = ['ground_truth', 'random', 'PCA', 'PCA_split', 'k_means', 'nmf', 'GNN']
    labels = ['Ground Truth', 'Random', 'PCA', 'PCA Split', 'K-means', 'NMF', 'GNN']

    ofile = osp.join(args.data_folder, 'max_ent_table.txt')
    with open(ofile, 'w') as o:
        o.write("\\begin{center}\n")
        o.write("\\begin{tabular}{|c|c|c|c|c|}\n")
        o.write("\\hline\n")
        o.write("Method & k & Pearson R & Avg Dist Pearson R & SCC \\\ \n")
        o.write("\\hline\\hline\n")
        for k in range(1, 10):
            empty = True # change to False if any methods are found for given k
            first = True # only write k for first row in section
            for method, label in zip(methods, labels):
                json_file = osp.join(args.sample_folder, method, 'k{}'.format(k), 'distance_pearson.json')
                if osp.exists(json_file):
                    empty = False
                    with open(json_file, 'r') as f:
                        results = json.load(f)
                        overall_pearson = results['overall_pearson']
                        scc = results['scc']
                        avg_dist_pearson = results['avg_dist_pearson']

                    if first:
                        k_label = k
                        first = False
                    else:
                        k_label = ''
                    o.write("{} & {} & {} &  {} & {} \\\ \n".format(label, k_label, overall_pearson, avg_dist_pearson, scc))
            if not empty:
                o.write("\\hline\n")
        o.write("\\end{tabular}\n")
        o.write("\\end{center}")

def main():
    args = getArgs()
    if args.sample is not None:
        makeLatexTableSample(args)
    if args.samples is not None:
        makeLatexTableSamples(args)

if __name__ == '__main__':
    main()
