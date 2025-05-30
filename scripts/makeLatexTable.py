import argparse
import json
import os
import os.path as osp
import re
import shutil
import sys
import tarfile
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from data_generation.modify_maxent import get_samples
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import (calculate_D, calculate_diag_chi_step,
                                      calculate_L, calculate_U)
from pylib.utils.load_utils import (get_converged_max_ent_folder,
                                    get_final_max_ent_folder, load_L,
                                    load_max_ent_chi, load_psi, load_Y)
from pylib.utils.similarity_measures import SCC, genome_disco, hic_spector
from pylib.utils.utils import (load_import_log, load_json, print_time,
                               triu_to_full)
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def load_time_dir(dir):
    assert osp.exists(dir), dir
    def load_time_file(file):
        if not osp.exists(file):
            print(f'Warning: {file} does not exist')
            return 0
        t = None
        with open(file) as f:
            for line in f:
                if line.startswith('elapsed'):
                    line_split = line.split()
                    t = int(line_split[1][:-3])
        return t

    eq_log_file = osp.join(dir, 'equilibration/log.log')
    if osp.exists(osp.join(dir, 'equilibration.tar.gz')):
        t_file = tarfile.open(osp.join(dir, 'equilibration.tar.gz'))
        try:
            log = t_file.extract('equilibration/log.log', dir)
        except KeyError:
            print(dir)
            raise

    t_eq = load_time_file(eq_log_file)

    if osp.exists(osp.join(dir, 'production_out/log.log')):
        t_prod = load_time_file(osp.join(dir, 'production_out/log.log'))
    elif osp.exists(osp.join(dir, 'production_out.tar.gz')):
        t_file = tarfile.open(osp.join(dir, 'production_out.tar.gz'))
        log = t_file.extract('production_out/log.log', dir)
        t_prod = load_time_file(osp.join(dir, 'production_out/log.log'))
    else:
        assert osp.exists(osp.join(dir, 'core0')), dir
        t_prod = []
        for f in os.listdir(dir):
            if f.startswith('core'):
                t_prod.append(load_time_file(osp.join(dir, f, 'log.log')))
        t_prod = np.sum(t_prod)


    # cleanup
    for t_file in ['equilibration.tar.gz', 'production_out.tar.gz']:
        if osp.exists(osp.join(dir, t_file)):
            file = t_file.split('.')[0]
            shutil.rmtree(osp.join(dir, file))

    return t_eq + t_prod

def getArgs(data_folder = None, sample = None, samples = None):
    parser = argparse.ArgumentParser(description='Base parser')

    parser.add_argument('--data_folder', type=str, default=data_folder,
                            help='location of input data')
    parser.add_argument('--sample', type=str, default=sample,
                            help='sample id')
    parser.add_argument('--samples', type=list, default=samples,
                            help='list of sample ids separated by -')
    parser.add_argument('--sample_folder', type=str,
                        help='location of input data')
    parser.add_argument('--experimental', action='store_true',
                        help="True for experimental data mode - ground truth won't be present")
    parser.add_argument('--convergence_definition', type=str,
                        help='key to define convergence {strict, normal, param} ("all" for all 3) (None to skip)')
    parser.add_argument('--convergence_mask', action='store_true',
                        help="True to mask samples for which max ent didn't converge")
    parser.add_argument('--gnn_id', type=list,
                        help="only consider given gnn_id")
    parser.add_argument('--bad_methods', type=list,
                        help="ignore methods matching an str in bad_methods")
    parser.add_argument('--verbose', type=bool, default=True)

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
    data = defaultdict(lambda: defaultdict(lambda : defaultdict(list)))
    # k, method, metric : list of values acros samples

    converged_mask = np.ones(len(args.samples)).astype(bool)

    bad_methods = ['angle', '_old', '0.006', '0.06', 'bound', 'init_diag']
    if args.bad_methods is not None:
        bad_methods.extend(args.bad_methods)
        print(f"bad methods: {bad_methods}")

    for i, sample in enumerate(args.samples):
        if args.verbose:
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
            ground_truth_S = calculate_U(L, D)
        ground_truth_y, ground_truth_ydiag = load_Y(sample_folder)
        ground_truth_y /= np.mean(ground_truth_y.diagonal())
        np.fill_diagonal(ground_truth_y, 1)
        read_count = np.sum(np.triu(ground_truth_y))
        ground_truth_meanDist = DiagonalPreprocessing.genomic_distance_statistics(ground_truth_y, 'prob')
        ground_truth_pcs = epilib.get_pcs(ground_truth_ydiag, 12, align = True).T
        # ground_truth_pcs_soren = epilib.get_sequences(ground_truth_y, 5,
        #                                             randomized=True).T

        for fname in os.listdir(sample_folder):
            temp_dict = defaultdict(lambda: np.NaN)
            fpath = osp.join(sample_folder, fname)
            # methods should be formatted such that method.split('-')[0] is in METHODS
            if not fname.startswith('optimize') or '-' not in fname:
                continue
            skip = False
            for bad_method in bad_methods:
                if fname.endswith(bad_method):
                    skip = True
            if skip:
                continue

            method = fname
            method_type = '-'.join(fname.split('-')[1:])
            if method_type.startswith('GNN'):
                GNN_ID = None
                type_split = re.split('[-_]', method_type)
                for subtype in type_split:
                    if subtype.startswith('GNN'):
                        GNN_ID = int(subtype[-3:])
                if args.gnn_id is not None and GNN_ID not in args.gnn_id:
                    continue
                if args.verbose:
                    print(fname)
                k = 0
                converged_it = None
                if 'max_ent' in method:
                    converged_path = osp.join(fpath, 'iteration20')
                    times = [0]
                else:
                    converged_path = fpath
                    times = [load_time_dir(fpath)]
                S_file = osp.join(fpath, 'S.npy')
                if osp.exists(S_file):
                    S = np.load(S_file)
                else:
                    print(f'S does not exists for {fpath}')
                    continue
            elif 'max_ent' in method:
                if args.verbose:
                    print(fname)
                config = load_json(osp.join(fpath, 'resources/config.json'))
                k = config['nspecies']
                # convergence
                convergence_file = osp.join(fpath, 'convergence.txt')
                if not osp.exists(convergence_file):
                    continue
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
                            config = load_json(config_file)
                            chis = np.array(config['chis'])
                            chis = chis[np.triu_indices(len(chis))] # grab upper triangle
                            diag_chis = np.array(config['diag_chis'])

                    params = np.concatenate((diag_chis, chis), axis = 1)
                    if args.verbose:
                        print(params, params.shape)

                    eps = 1e-2
                    for j in range(1, len(params)):
                        diff = params[j] - params[j-1]
                        if np.linalg.norm(diff, ord = 2) < eps:
                            converged_it = j
                            break
                else:
                    # loss convergence
                    converged_it = get_converged_max_ent_folder(fpath,
                                                                args.convergence_definition,
                                                                throw_exception = False,
                                                                return_it = True)
                temp_dict['converged_it'] = converged_it
                if converged_it is not None:
                    converged_path = osp.join(fpath, f'iteration{converged_it+1}')
                    if not osp.exists(converged_path):
                        converged_path = get_final_max_ent_folder(fpath)
                else:
                    if args.verbose:
                        print(f'\tDID NOT CONVERGE: {conv}')
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

                    config = load_json(osp.join(converged_path, 'config.json'))
                    diag_chis_continuous = calculate_diag_chi_step(config)
                    D = calculate_D(diag_chis_continuous)

                    # calc S
                    S = calculate_U(L, D)
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
                    print(f'y_file not found')
                    # presumably max ent is still running
                    continue
                yhat_meanDist = DiagonalPreprocessing.genomic_distance_statistics(yhat)
                yhat_diag = DiagonalPreprocessing.process(yhat, yhat_meanDist, verbose = False)
                scc = SCC(h=5, K=100)
                corr_scc_var = scc.scc(ground_truth_y, yhat)

                # result = plotDistanceStratifiedPearsonCorrelation(ground_truth_y,
                                # yhat, converged_path)
                # overall_corr, corr_scc, corr_scc_var, avg_diag = result
                # temp_dict['overall_pearson'] = overall_corr
                # temp_dict['scc'] = corr_scc
                temp_dict['scc_var'] = corr_scc_var
                # temp_dict['avg_dist_pearson'] = avg_diag

                corr_hic_spector = hic_spector(ground_truth_y, yhat, 10)
                temp_dict['hic_spector'] = corr_hic_spector

                # corr_genome_disco = genome_disco(ground_truth_y, yhat, 3)
                # temp_dict['genome_disco'] = corr_genome_disco

                # rmse-diag
                rmse_diag = mean_squared_error(ground_truth_meanDist, yhat_meanDist,
                                                squared = False)
                temp_dict['rmse-diag'] = rmse_diag
                rmse_diag10 = mean_squared_error(ground_truth_meanDist[:10],
                                                yhat_meanDist[:10], squared = False)
                temp_dict['rmse-diag10'] = rmse_diag10
            else:
                print(f'converged path is None: {method}')
                yhat = None

            if ground_truth_S is not None and S is not None:
                rmse_S = mean_squared_error(ground_truth_S, S, squared = False)
                temp_dict['rmse-S'] = rmse_S

            rmse_y = None
            pearson_pc_1 = None
            pearson_pc_1_soren = None
            if yhat is not None:
                # normalize yhat
                yhat = yhat.astype(np.float64) / np.mean(np.diagonal(yhat))
                rmse_y = mean_squared_error(ground_truth_y, yhat, squared = False)
                temp_dict['rmse-y'] = rmse_y

                rmse_ydiag = mean_squared_error(ground_truth_ydiag, yhat_diag, squared=False)
                temp_dict['rmse-ydiag'] = rmse_ydiag

                pcs = epilib.get_pcs(yhat_diag, 12, align = True).T
                pearson_pc_1, _ = pearsonr(pcs[0], ground_truth_pcs[0])
                pearson_pc_1 *= np.sign(pearson_pc_1) # ensure positive pearson
                assert pearson_pc_1 > 0
                temp_dict['pearson_pc_1'] = pearson_pc_1

                # pcs_soren = epilib.get_sequences(yhat, 5, randomized=True).T
                # pearson_pc_1_soren, _ = pearsonr(pcs_soren[0], ground_truth_pcs_soren[0])
                # pearson_pc_1_soren *= np.sign(pearson_pc_1_soren) # ensure positive pearson
                # assert pearson_pc_1_soren > 0
                # temp_dict['pearson_pc_1_soren'] = pearson_pc_1_soren


            # append temp_dict to data
            data[k][method]['read_count'].append(read_count/1000)
            data[k][method]['scc_var'].append(temp_dict['scc_var'])
            data[k][method]['hic_spector'].append(temp_dict['hic_spector'])
            data[k][method]['genome_disco'].append(temp_dict['genome_disco'])
            data[k][method]['avg_dist_pearson'].append(temp_dict['avg_dist_pearson'])
            data[k][method]['rmse-S'].append(temp_dict['rmse-S'])
            data[k][method]['rmse-y'].append(temp_dict['rmse-y'])
            data[k][method]['rmse-ydiag'].append(temp_dict['rmse-ydiag'])
            data[k][method]['pearson_pc_1'].append(temp_dict['pearson_pc_1'])
            data[k][method]['pearson_pc_1_soren'].append(temp_dict['pearson_pc_1_soren'])
            data[k][method]['rmse-diag'].append(temp_dict['rmse-diag'])
            data[k][method]['rmse-diag10'].append(temp_dict['rmse-diag10'])
            data[k][method]['converged_time'].append(temp_dict['converged_time'])
            data[k][method]['total_time'].append(temp_dict['total_time'])
            data[k][method]['converged_it'].append(temp_dict['converged_it'])


    print(f'{np.sum(converged_mask)} out of {len(converged_mask)} converged')
    print(converged_mask)
    return data, converged_mask

def makeLatexTable(data, ofile, header, small, mode='w', sample_id=None,
                    experimental=False, nan_mask=None, test_significance=True):
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
        test_significance: True to perform two-sided paired t-tests for significance
    '''

    metric_labels = {'scc':'SCC', 'scc_var':'SCC', None: '', 'k': 'k',
                    'hic_spector': 'HiC-Spector', 'genome_disco': 'GenomeDISCO',
                    'rmse-S':'RMSE-Energy', 'rmse-y':'RMSE(H)', 'rmse-ydiag':r'RMSE($\tilde{H}$)',
                    'pearson_pc_1':'Corr PC 1', 'pearson_pc_1_soren':'Corr Soren PC 1',
                    'rmse-diag':'RMSE-P(s)', 'rmse-diag10':'RMSE-P(s<10)',
                    'avg_dist_pearson':'SCC mean', 'read_count':'Read Count (1k)',
                    'total_time':'Total Time', 'converged_it':'Converged It.',
                    'converged_time':'Simulation Time', 'prcnt_converged': '\% Converged'}
    for k, v in metric_labels.items():
        metric_labels[k] = r'\thead{' + v + '}'
    if small:
        metrics = ['scc_var', 'hic_spector', 'pearson_pc_1', 'rmse-ydiag', 'converged_time']
        # 'rmse-diag',
    else:
        metrics = ['rmse-y', 'rmse-ydiag',  'converged_time', 'converged_it', 'prcnt_converged']


    print(f'\nMAKING TABLE-small={small}')
    header = header.replace('_', "\_")
    with open(ofile, mode) as o:
        # set up first rows of table
        o.write("\\begin{center}\n")
        if experimental and 'rmse-S' in metrics:
            metrics.remove('rmse-S')
        num_cols = len(metrics) + 1
        num_cols_str = str(num_cols)

        o.write("\\begin{tabular}{|" + "c|"*num_cols + "}\n")
        o.write("\\hline\n")
        o.write("\\multicolumn{" + num_cols_str + "}{|c|}{" + header + "} \\\ \n")
        if sample_id is not None:
            o.write("\\hline\n")
            o.write("\\multicolumn{" + num_cols_str + "}{|c|}{sample " + f'{sample_id}' + "} \\\ \n")

        o.write("\\hline\n")

        row = "Method"
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

        GNN_ref_method = None
        if 0 in data.keys():
            for method in sorted(data[0].keys()):
                if 'GNN' in method:
                    GNN_ref_method = method
                    break # takes first GNN found
        # GNN_ref_method = 'optimize_grid_b_200_v_8_spheroid_1.5-GNN673'

        if GNN_ref_method is not None:
            GNN_ref = data[0][GNN_ref_method]
            print(f'GNN found: using GNN {GNN_ref_method}')
            print(GNN_ref)
        else:
            GNN_ref = None

        for k in sorted(data.keys()):
            first = True # only write k for first row in section
            keys_labels = sort_method_keys(data[k].keys())
            for method, label in keys_labels:
                dataset = None
                if 'GNN' in method and 'max_ent' not in method:
                    pos = method.find('GNN')
                    id = method[pos+3:]
                    id = id.split('-')[0]
                    try:
                        argparse_file = osp.join('/home/erschultz/sequences_to_contact_maps/',
                                                f'results/ContactGNNEnergy/{id}/argparse.txt')
                        with open(argparse_file, 'r') as f:
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
                if nan_mask is not None and len(data[k][method]['scc_var']) != len(nan_mask):
                    # skip method if not performed for all samples
                    print(f'skipping {method}, k={k}: {data[k][method]["scc_var"]}')
                    continue

                if dataset is None:
                    text = f"{label}"
                else:
                    text = f"{label} ({dataset})"

                if 'k' in metrics:
                    if first: # only write k for first row in section
                        k_label = k
                        if k_label == 0:
                            k_label = '-'
                        first = False
                    else:
                        k_label = ''
                    text += f" & {k_label}"

                for metric in metrics:
                    significant = False # two sided t test
                    if metric == 'prcnt_converged':
                        converged_it = data[k][method]['converged_it']
                        not_converged = 0
                        for i in converged_it:
                            if i is None:
                                not_converged += 1
                        result = 1 - not_converged / len(converged_it)
                        result *= 100
                        result = np.array([result])
                    elif metric == 'k':
                        continue
                    elif metric is None:
                        result = np.array([np.NaN])
                    else:
                        result = np.array(data[k][method][metric], dtype=np.float64)
                        if nan_mask is not None:
                            result[nan_mask] = np.NaN

                    roundoff = 2
                    if metric is None:
                        pass # metric is None for filler column
                    elif 'time' in metric or metric == 'read_count':
                        roundoff = 1
                    elif metric == 'rmse-S':
                        roundoff = 2
                    elif metric == 'rmse-y' or metric.startswith('rmse-diag'):
                        roundoff = 5

                    # if metric == 'scc_var':
                    #     print('scc_var: ')
                    #     print(method)
                    #     print(result)

                    if len(result) > 1:
                        result_mean = np.nanmean(result)
                        # print(metric, result, result_mean)
                        if GNN_ref is not None and test_significance:
                            ref_result = np.array(GNN_ref[metric], dtype=np.float64)
                            if nan_mask is not None:
                                try:
                                    ref_result[nan_mask] = np.nan
                                except IndexError:
                                    print(nan_mask.shape)
                                    print(metric)
                                    print(ref_result.shape)
                                    raise
                            stat, pval = ss.ttest_rel(ref_result, result, nan_policy='omit')
                            mean_effect_size = np.mean(result - ref_result)
                            mean_effect_size = np.round(mean_effect_size, 3)
                            if pval < 0.05:
                                print('Significant:', metric, pval, mean_effect_size)
                        else:
                            pval = None
                        result_mean = np.round(result_mean, roundoff)
                        result_std = np.nanstd(result, ddof=1)
                        result_sem = ss.sem(result, nan_policy = 'omit')
                        result_std = np.round(result_std, roundoff)
                        result_sem = np.round(result_sem, roundoff)
                        text += f" & {result_mean} $\pm$ {result_sem}"
                        # text += f" & {result_mean}"
                        if pval is None:
                            pass
                        elif pval < 0.001:
                            text += f' ***'
                        elif pval < 0.01:
                            text += f' **'
                        elif pval < 0.05:
                            text += f' *'
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
            if substr == 'v':
                v = key_split[i+1]
            if substr.startswith('angle'):
                angle = substr[5:]
            if substr.startswith('spheroid'):
                ar = key_split[i+1]

        if 'GNN' in key:
            pos = key.find('GNN')
            id = key[pos+3:]
            label += id

        if '_distance_' in key:
            label += ' (distance)'

        # if 'phi' in key:
        #     label += f' (b{b}, $\phi${phi}'
        # elif 'v' in key:
        #     label += f' (b{b}, v{v}'
        #
        # if 'angle' in key:
        #     label += f', angle{angle})'
        # elif 'spheroid' in key:
        #     label += f', ar{ar})'
        # else:
        #     label += ')'

        key_split = key.split('-')
        other = key_split[-1].split('_')
        if len(key_split) == 3:
            other = [key_split[-2]] + other
        if len(other) > 2:
            label += ' '
            label += '\_'.join(other[2:])

        return label

    gnn_key_labels = [] # list of (key, label) tuples of type method
    me_key_labels = []
    for key in keys:
        if 'max_ent' in key.lower():
            me_key_labels.append((key, format(key, 'Max Ent')))
        elif 'gnn' in key.lower():
            me_key_labels.append((key, format(key, 'GNN')))

    sorted_key_labels.extend(sorted(gnn_key_labels))
    sorted_key_labels.extend(sorted(me_key_labels))

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

def main(args=None, fname='max_ent_table.txt'):
    if args is None:
        args = getArgs()
    assert args.sample is None or args.samples is None
    print('-'*100, '\n', args)

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
                        nan_mask = not_converged_mask,
                        test_significance = args.test_significance)
        mode = 'a' # switch to append for remaning tables

        makeLatexTable(data, ofile, temp_label, False, mode,
                        sample_id = args.sample, experimental = args.experimental,
                        nan_mask = not_converged_mask,
                        test_significance = args.test_significance)

        # boxplot(data, osp.join(odir, f'boxplot_{defn}_convergence.png'))

def main_k():
    dataset = 'dataset_12_06_23'
    samples, _ = get_samples(dataset, train=True, filter_cell_lines=['imr90'])
    samples_list = samples[:10]
    print(samples_list)

    args = getArgs(data_folder = f'/home/erschultz/{dataset}',
                    samples = samples_list)
    args.experimental = True
    args.verbose = True
    args.bad_methods=['phi', 'grid200', '_spheroid_2.0']
    args.convergence_definition = 'normal'
    args.gnn_id = []
    args.test_significance = False
    main(args)


def main_mouse():
    dataset = 'dataset_mouse_50k_512'
    samples, _ = get_samples(dataset, test = True,
                            filter_cell_lines=['ch12-lx-b-lymphoblasts'])
    print(samples, len(samples))

    data_dir = osp.join('/project/depablo/erschultz', dataset)
    # data_dir = osp.join('/home/erschultz', dataset)
    args = getArgs(data_folder = data_dir, sample = None, samples = samples)
    args.experimental = True
    args.verbose = True
    args.convergence_definition = 'normal'
    args.test_significance = True
    args.bad_methods = ['_stop', 'b_140', 'b_261', 'spheroid_2.0', '_700k', 'phi',
                        'GNN579-max_ent', '-gd_gamma', 'distance', 'start', 'stat',
                        'diagbins', 'binarize', 'chrom', 'grid200', 'long', 'long5',
                        '_repeat-GNN690', '_test', '_repeat', 'max_ent10-GNN690',
                        '_chipseq', '_repeat2']
    for i in [1,2,3,4,5,7,8,9,10, 11,12,13,14,15]:
       args.bad_methods.append(f'max_ent{i}')
    args.gnn_id = [690]
    main(args)

def main_human():
    sample=None
    dataset='dataset_12_06_23'
    # dataset='Su2020'; samples = [1013]
    all_cell_lines = ['imr90', 'hap1', 'huvec', 'gm12878', 'hmec']

    for cell_line in all_cell_lines:
        samples, _ = get_samples(dataset, test = True,
                                    filter_cell_lines=[cell_line])
        print(samples, len(samples))

        data_dir = osp.join('/project/depablo/erschultz', dataset)
        # data_dir = osp.join('/home/erschultz', dataset)
        args = getArgs(data_folder = data_dir, sample = sample, samples = samples)
        args.experimental = True
        args.verbose = True
        args.convergence_definition = 'strict'
        args.test_significance = True
        args.bad_methods = ['_stop', 'b_140', 'b_261', 'spheroid_2.0', '_700k', 'phi',
                            'GNN579-max_ent', '-gd_gamma', 'distance', 'start', 'stat',
                            'diagbins', 'binarize', 'chrom', 'grid200', 'long', 'long5',
                            '_repeat-GNN690', '_test', '_repeat', 'max_ent10-GNN690',
                            '_chipseq', '_repeat2']
        for i in [1,2,3,4,5,7,8,9,10, 11,12,13,14,15]:
           args.bad_methods.append(f'max_ent{i}')
        args.gnn_id = [690]
        # args.gnn_id = []
        main(args, fname=f'max_ent_table_{cell_line}.txt')
        # data, converged_mask = load_data(args)
        # boxplot(data, osp.join(data_dir, 'boxplot_test.png'))
        return


if __name__ == '__main__':
    main_k()
    # main_mouse()
    # main_human()
