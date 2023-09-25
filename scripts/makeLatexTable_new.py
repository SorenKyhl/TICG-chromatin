import argparse
import json
import os
import os.path as osp
import re
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from compare_contact import plotDistanceStratifiedPearsonCorrelation
from data_generation.modify_maxent import get_samples
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import (calculate_D, calculate_diag_chi_step,
                                      calculate_L, calculate_S)
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import triu_to_full
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import ArgparserConverter
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_import_log, load_L, load_max_ent_chi,
    load_psi, load_Y)
from sequences_to_contact_maps.scripts.utils import load_time_dir, print_time


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
    parser.add_argument('--experimental', action='store_true',
                        help="True for experimental data mode - ground truth won't be present")
    parser.add_argument('--convergence_definition', type=AC.str2None,
                        help='key to define convergence {strict, normal, param} ("all" for all 3) (None to skip)')
    parser.add_argument('--convergence_mask', action='store_true',
                        help="True to mask samples for which max ent didn't converge")
    parser.add_argument('--gnn_id', type=AC.str2list,
                        help="only consider given gnn_id")

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

def load_data(args):
    '''
    Loads data from args.data_folder for each sample <i> in args.samples.

    Inputs:
        args: parsed command line arguments
    Outputs:
        data: nested dictionary
        converged_mask: mask for list of replicate arrays, 1 if ALL max ent runs converged
    '''
    print(f'\nLOADING DATA with {args.convergence_definition} convergence')
    data = defaultdict(lambda: defaultdict(lambda : defaultdict(list))) # k, method, metric : list of values acros samples
    converged_mask = np.ones(len(args.samples)).astype(bool)

    for i, sample in enumerate(args.samples):
        print(i, sample)
        sample_folder = osp.join(args.data_folder, 'samples', f'sample{sample}')
        if args.experimental:
            ground_truth_S = None
        else:
            diag_chis_file = osp.join(sample_folder, 'diag_chis_continuous.npy')
            if not osp.exists(diag_chis_file):
                diag_chi_continuous = np.load(osp.join(sample_folder, 'diag_chis.npy'))
            else:
                diag_chi_continuous = np.load(diag_chis_file)
            D = calculate_D(diag_chi_continuous)
            L = load_L(sample_folder, save = True)
            ground_truth_S = calculate_S(L, D)
        ground_truth_y, ground_truth_ydiag = load_Y(sample_folder)
        ground_truth_meanDist = DiagonalPreprocessing.genomic_distance_statistics(ground_truth_y, 'prob')
        ground_truth_pcs = epilib.get_pcs(ground_truth_ydiag, 12, align = True).T

        for fname in os.listdir(sample_folder):
            temp_dict = defaultdict(lambda: np.NaN)
            fpath = osp.join(sample_folder, fname)
            # methods should be formatted such that method.split('-')[0] is in METHODS
            if not fname.startswith('optimize') or '-' not in fname:
                continue
            bad_methods = ['angle', '_old', '0.006', '0.06', 'bound', 'init_diag', "_repeat"]
            skip = False
            for bad_method in bad_methods:
                if bad_method in fname:
                    skip = True
            if skip:
                continue

            method = fname
            method_type = fname.split('-')[1]
            if method_type.startswith('GNN'):
                GNN_ID = int(fname.strip('_repeat')[-3:])
                if args.gnn_id is not None and GNN_ID not in args.gnn_id:
                    continue
                print(fname)
                k = 0
                converged_it = None
                converged_path = fpath
                S = np.load(osp.join(fpath, 'S.npy'))
                times = [load_time_dir(fpath)]
            elif method_type.startswith('max_ent'):
                print(fname)
                chi_path = osp.join(fpath, 'chis.txt')
                if not osp.exists(chi_path):
                    # presumabely it is still running
                    continue
                k = len(triu_to_full(np.loadtxt(chi_path)))

                # convergence
                convergence_file = osp.join(fpath, 'convergence.txt')
                assert osp.exists(convergence_file)
                conv = np.atleast_1d(np.loadtxt(convergence_file))
                converged_it = None
                if args.convergence_definition is None:
                    _, converged_it = get_final_max_ent_folder(fpath, return_it = True)
                elif args.convergence_definition == 'param':
                    # param convergence
                    all_chis = []
                    all_diag_chis = []
                    for i in range(30):
                        it_path = osp.join(fpath, f'iteration{i}')
                        if osp.exists(it_path):
                            config_file = osp.join(it_path, 'production_out/config.json')
                            config = json.load(config_file)
                            chis = np.array(config['chis'])
                            chis = chis[np.triu_indices(len(chis))] # grab upper triangle
                            diag_chis = np.array(config['diag_chis'])

                    params = np.concatenate((diag_chis, chis), axis = 1)
                    print(params, params.shape)

                    eps = 1e-2
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
                        raise Exception('Unrecognized convergence_definition: '
                                        f'{args.convergence_definition}')


                    for j in range(1, len(conv)):
                        diff = conv[j] - conv[j-1]
                        if np.abs(diff) < eps and conv[j] < conv[0]:
                            converged_it = j
                            break
                temp_dict['converged_it'] = converged_it
                if converged_it is not None:
                    converged_path = osp.join(fpath, f'iteration{converged_it}')
                else:
                    print('\tDID NOT CONVERGE')
                    converged_mask[i] = 0
                    converged_path = None

                # times
                its = len(conv)
                times = []
                for it in range(its):
                    times.append(load_time_dir(osp.join(fpath, f'iteration{it}')))

                # S
                if converged_it is not None:
                    # load bead types
                    psi = load_psi(fpath)

                    # load chi
                    chi = load_max_ent_chi(k, converged_path, throw_exception = True)

                    # calc s
                    L = calculate_L(psi, chi)

                    with open(osp.join(converged_path, 'config.json'), 'r') as f:
                        config = json.load(f)
                    diag_chis_continuous = calculate_diag_chi_step(config)
                    D = calculate_D(diag_chis_continuous)

                    # calc S
                    S = calculate_S(L, D)
                else:
                    S = None
            else:
                # method must be GNN or max_ent
                continue


            # time
            if converged_it is None:
                converge_time = np.sum(times) / 60
            else:
                converge_time = np.sum(times[:converged_it])
                converge_time /= 60 # to minutes
            temp_dict['converged_time'] = converge_time
            temp_dict['total_time'] = np.sum(times) / 60

            if converged_path is not None:
                # SCC
                data_file = osp.join(converged_path, 'production_out')
                y_file = osp.join(converged_path, 'y.npy')
                y_file2 = osp.join(converged_path, 'production_out/contacts.txt')
                if osp.exists(y_file):
                    yhat = np.load(y_file)
                elif osp.exists(y_file2):
                    yhat = np.loadtxt(y_file2)
                else:
                    # presumably max ent is still running
                    continue
                yhat_meanDist = DiagonalPreprocessing.genomic_distance_statistics(yhat)
                yhat_diag = DiagonalPreprocessing.process(yhat, yhat_meanDist, verbose = False)
                scc = SCC()
                corr_scc_var = scc.scc(ground_truth_ydiag, yhat_diag, var_stabilized = True)
                print(corr_scc_var)
                # result = plotDistanceStratifiedPearsonCorrelation(ground_truth_y,
                                # yhat, converged_path)
                # overall_corr, corr_scc, corr_scc_var, avg_diag = result
                # temp_dict['overall_pearson'] = overall_corr
                # temp_dict['scc'] = corr_scc
                temp_dict['scc_var'] = corr_scc_var
                # temp_dict['avg_dist_pearson'] = avg_diag

                # rmse-diag
                rmse_diag = mean_squared_error(ground_truth_meanDist, yhat_meanDist, squared = False)
                temp_dict['rmse-diag'] = rmse_diag
            else:
                yhat = None

            if ground_truth_S is not None and S is not None:
                rmse_S = mean_squared_error(ground_truth_S, S, squared = False)
                temp_dict['rmse-S'] = rmse_S

            rmse_y = None
            pearson_pc_1 = None
            if yhat is not None:
                # normalize yhat
                yhat = yhat.astype(np.float64) / np.mean(np.diagonal(yhat))
                rmse_y = mean_squared_error(ground_truth_y, yhat, squared = False)
                temp_dict['rmse-y'] = rmse_y

                pcs = epilib.get_pcs(yhat_diag, 12, align = True).T
                pearson_pc_1, _ = pearsonr(pcs[0], ground_truth_pcs[0])
                pearson_pc_1 *= np.sign(pearson_pc_1) # ensure positive pearson
                assert pearson_pc_1 > 0
                temp_dict['pearson_pc_1'] = pearson_pc_1


            # append temp_dict to data
            data[k][method]['overall_pearson'].append(temp_dict['overall_pearson'])
            data[k][method]['scc'].append(temp_dict['scc'])
            data[k][method]['scc_var'].append(temp_dict['scc_var'])
            data[k][method]['avg_dist_pearson'].append(temp_dict['avg_dist_pearson'])
            data[k][method]['rmse-S'].append(temp_dict['rmse-S'])
            data[k][method]['rmse-y'].append(temp_dict['rmse-y'])
            data[k][method]['pearson_pc_1'].append(temp_dict['pearson_pc_1'])
            data[k][method]['rmse-diag'].append(temp_dict['rmse-diag'])
            data[k][method]['converged_time'].append(temp_dict['converged_time'])
            data[k][method]['total_time'].append(temp_dict['total_time'])
            data[k][method]['converged_it'].append(temp_dict['converged_it'])


    print(f'{np.sum(converged_mask)} out of {len(converged_mask)} converged')
    print(converged_mask)
    return data, converged_mask

def makeLatexTable(data, ofile, header, small, mode = 'w', sample_id = None,
                    experimental = False, nan_mask = None):
    '''
    Writes data to ofile in latex table format.

    Inputs:
        data: dictionary containing data from load_data
        ofile: location to write data
        small: True to output smaller table with fewer metrics')
        mode: file mode (e.g. 'w', 'a')
        sample_id: sample_id for table header; if set, table will show stdev over replicates for that sample
        experimental: True if ground_truth won't be present
        na_mask: mask for sample_results, True to set to nan
    '''

    metric_labels = {'scc':'SCC', 'scc_var':'SCC var',
                    'rmse-S':'RMSE-Energy', 'rmse-y':'RMSE-Y', 'pearson_pc_1':'Corr PC 1',
                    'rmse-diag':'RMSE-P(s)', 'avg_dist_pearson':'SCC mean',
                    'total_time':'Total Time', 'converged_it':'Converged It.',
                    'converged_time':'Converged Time'}
    if small:
        metrics = ['scc_var', 'rmse-diag', 'pearson_pc_1', 'converged_time']
    else:
        metrics = ['converged_it', 'converged_time', 'total_time']


    print(f'\nMAKING TABLE-small={small}')
    header = header.replace('_', "\_")
    with open(ofile, mode) as o:
        # set up first rows of table
        o.write("\\begin{center}\n")
        if experimental and 'rmse-S' in metrics:
            metrics.remove('rmse-S')
        num_cols = len(metrics) + 2
        num_cols_str = str(num_cols)

        o.write("\\begin{tabular}{|" + "c|"*num_cols + "}\n")
        o.write("\\hline\n")
        o.write("\\multicolumn{" + num_cols_str + "}{|c|}{" + header + "} \\\ \n")
        if sample_id is not None:
            o.write("\\hline\n")
            o.write("\\multicolumn{" + num_cols_str + "}{|c|}{sample " + f'{sample_id}' + "} \\\ \n")

        o.write("\\hline\n")

        row = "Method & k"
        for metric in metrics:
            label = metric_labels[metric]
            row += f" & {label}"
        row += ' \\\ \n'
        o.write(row)
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
        GNN_ref_method = None
        if 0 in data.keys():
            for method in sorted(data[0].keys()):
                if 'GNN' in method:
                    GNN_ref = data[0][method]
                    GNN_ref_method = method
                    break # takes first GNN found

        if GNN_ref is not None:
            print(f'GNN found: using GNN {GNN_ref_method}')
            print(GNN_ref)

        for k in sorted(data.keys()):
            first = True # only write k for first row in section
            keys_labels = sort_method_keys(data[k].keys())
            for method, label in keys_labels:
                dataset = None
                if 'GNN' in method:
                    pos = method.find('GNN')
                    id = method[pos+3:]
                    try:
                        with open(f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{id}/argparse.txt', 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line == '--data_folder':
                                    dataset = f.readline().strip()
                                    dataset_list = dataset.split('-')
                                    dataset_list = [osp.split(i)[1] for i in dataset_list]
                                    dataset_list = [i[8:].replace('_', '/') for i in dataset_list]
                                    dataset = '-'.join(dataset_list)
                                    break
                    except Exception as e:
                        dataset = None
                        print(e)
                if nan_mask is not None and len(data[k][method]['scc']) != len(nan_mask):
                    # skip method if not performed for all samples
                    continue
                if first: # only write k for first row in section
                    k_label = k
                    if k_label == 0:
                        k_label = '-'
                    first = False
                else:
                    k_label = ''
                if dataset is None:
                    text = f"{label} & {k_label}"
                else:
                    text = f"{label} ({dataset}) & {k_label}"

                for metric in metrics:
                    significant = False # two sided t test
                    result = np.array(data[k][method][metric], dtype=np.float64)


                    if nan_mask is not None:
                        result[nan_mask] = np.NaN

                    if 'time' in metric:
                        roundoff = 1
                    elif metric == 'rmse-S':
                        roundoff = 2
                    elif metric == 'rmse-y' or metric == 'rmse-diag':
                        roundoff = 4
                    else:
                        roundoff = 3

                    if len(result) > 1:
                        result_mean = np.nanmean(result)
                        # print(metric, result, result_mean)
                        ref_result = np.array(GNN_ref[metric], dtype=np.float64)
                        if nan_mask is not None:
                            ref_result[nan_mask] = np.nan
                        stat, pval = ss.ttest_rel(ref_result, result)
                        if pval < 0.05:
                            print('Significant:', metric, pval)
                            significant = True

                        result_mean = np.round(result_mean, roundoff)
                        result_std = np.nanstd(result, ddof=1)
                        result_sem = ss.sem(result, nan_policy = 'omit')
                        result_std = np.round(result_std, roundoff)
                        result_sem = np.round(result_sem, roundoff)
                        text += f" & {result_mean} $\pm$ {result_sem}"
                        if significant:
                            text += f' *{np.round(pval, 5)}'
                    else:
                        result = result[0]
                        if result is not None:
                            result = np.round(result, roundoff)
                        text += f" & {result}"

                text += " \\\ \n"

                o.write(text)
            o.write("\\hline\n")
        o.write("\\end{tabular}\n")
        o.write("\\end{center}\n\n")

def sort_method_keys(keys):
    '''Sorts keys to match order of METHODS and gets corresponding labels from LABELS.'''
    sorted_key_labels = []
    def format(key, type):
        label = type
        key_split = re.split('[-_]', key)
        for i, substr in enumerate(key_split):
            if substr == 'b':
                b = key_split[i+1]
            if substr == 'phi':
                phi = key_split[i+1]
            if substr.startswith('angle'):
                angle = substr[5:]
            if substr.startswith('spheroid'):
                ar = key_split[i+1]

        if 'GNN' in key:
            pos = key.find('GNN')
            id = key[pos+3:]
            label += id

        if 'angle' in key:
            label += f' (b={b},phi={phi}, angle{angle})'
        elif 'spheroid' in key:
            label += f' (b={b},phi={phi}, ar{ar})'
        else:
            label += f'  (b={b},phi={phi})'

        left, right = key.split('-')
        other = right.split('_')
        if len(other) > 2:
            label += ' '
            label += '\_'.join(other[2:])

        return label

    for method, label in zip(['max_ent', 'gnn'], ['Max Ent', 'GNN']):
        key_labels = [] # list of (key, label) tuples of type method
        for key in keys:
            if method in key.lower():
                key_labels.append((key, format(key, label)))

        sorted_key_labels.extend(sorted(key_labels))

    return sorted_key_labels

def boxplot(data, ofile):
    max_ent = 'optimize_grid_b_140_phi_0.03-max_ent10'
    max_ent_times = data[10][max_ent]['converged_time']
    max_ent_times = [i for i in max_ent_times if i is not None]
    max_ent_sccs = data[10][max_ent]['scc_var']
    max_ent_sccs = [i for i in max_ent_sccs if not np.isnan(i)]

    gnn = 'GNN403'
    gnn_times = data[0][gnn]['total_time']
    gnn_sccs = data[0][gnn]['scc_var']

    labels = ['Max. Ent.', 'GNN']

    fig, axes = plt.subplots(1, 2)
    data = [max_ent_sccs, gnn_sccs]
    print('scc data', data)
    b1 = axes[0].boxplot(data, vert = True,
                        patch_artist = True, labels = labels)
    axes[0].set_ylabel('SCC', fontsize=16)

    data = [max_ent_times, gnn_times]
    print('time data', data)
    b2 = axes[1].boxplot(data,  vert = True,
                        patch_artist = True, labels = labels)
    # axes[1].set_yscale('log')
    axes[1].set_ylabel('Time (mins)', fontsize=16)

    # fill with colors
    colors = ['b' ,'r']
    for bplot in [b1, b2]:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig(ofile)



def main(args=None):
    if args is None:
        args = getArgs()
    assert args.sample is None or args.samples is None
    print('-'*100, '\n', args)

    fname = 'max_ent_table.txt'
    dataset = osp.split(args.data_folder)[1]
    label = dataset


    if args.sample is not None:
        args.samples = [args.sample]
        odir = args.sample_folder
    else:
        odir = args.data_folder
    ofile = osp.join(odir, fname)

    if args.convergence_definition == 'all':
        convergence_def_list = ['normal', 'strict', None]
    elif args.convergence_definition == 'both':
        convergence_def_list = ['normal', None]
    else:
        convergence_def_list = [args.convergence_definition]

    mode = 'w'
    for defn in convergence_def_list:
        args.convergence_definition = defn
        temp_label = label + f' {defn} convergence'
        data, converged_mask = load_data(args)
        with open(osp.join(odir, f'data_{defn}_convergence.json'), 'w') as f:
            json.dump(data, f)

        print('Data loaded')
        not_converged_mask = np.logical_not(converged_mask) # now 1 if not converged
        if not args.convergence_mask:
            # manually override
            not_converged_mask = np.zeros_like(not_converged_mask)
        else:
            temp_label += ' (only converged)'

        # small
        makeLatexTable(data, ofile, temp_label, True, mode,
                        sample_id = args.sample, experimental = args.experimental,
                        nan_mask = not_converged_mask)
        mode = 'a' # switch to append for remaning tables

        makeLatexTable(data, ofile, temp_label, False, mode,
                        sample_id = args.sample, experimental = args.experimental,
                        nan_mask = not_converged_mask)

        # boxplot(data, osp.join(odir, f'boxplot_{defn}_convergence.png'))


if __name__ == '__main__':
    dataset = 'dataset_02_04_23'
    # dataset = 'dataset_09_17_23'
    # dataset = 'dataset_08_25_23'; samples = range(1, 16)
    # dataset = 'dataset_04_28_23'; samples = [1,2,3,4,5,324,981,1753,1936,2834,3464]
    # dataset = 'dataset_06_29_23'; samples = [1,2,3,4,5,101,102,103,104,105,601,602,603,604,605]
    # dataset='Su2020'; samples = [1013]

    samples, _ = get_samples(dataset, test = True)
    samples = samples[:10]

    data_dir = osp.join('/home/erschultz', dataset)
    args = getArgs(data_folder = data_dir, samples = samples)
    args.experimental = True
    args.convergence_definition = 'normal'
    args.gnn_id=[434, 451, 461, 462, 471, 472, 476, 477, 479, 480, 481, 485]
    args.gnn_id=[434, 451, 455, 456, 461, 462, 463, 470, 471, 472, 476, 477, 479, 480, 481, 484, 485, 486, 488]

    args.gnn_id=[485]
    main(args)
    # data, converged_mask = load_data(args)
    # boxplot(data, osp.join(data_dir, 'boxplot_test.png'))
