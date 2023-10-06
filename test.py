import csv
import json
import math
import os
import os.path as osp
import shutil
import sys
import tarfile
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import sympy
import torch
import torch_geometric
from pylib.utils import default, epilib, hic_utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import (calculate_all_energy, calculate_D,
                                      calculate_diag_chi_step, calculate_S)
from pylib.utils.plotting_utils import BLUE_RED_CMAP
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import load_json, pearson_round, triu_to_full
from pylib.utils.xyz import xyz_load, xyz_to_distance
from scipy.ndimage import uniform_filter
from scripts.data_generation.modify_maxent import get_samples
from scripts.get_params import GetEnergy
from scripts.makeLatexTable_new import getArgs, load_data
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')

from sequences_to_contact_maps.scripts.knightRuiz import knightRuiz
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_all, load_contact_map, load_max_ent_D,
    load_max_ent_L, load_Y)
from sequences_to_contact_maps.scripts.plotting_utils import (plot_diag_chi,
                                                              plot_matrix,
                                                              plot_seq_binary)
from sequences_to_contact_maps.scripts.R_pca import R_pca
from sequences_to_contact_maps.scripts.utils import rescale_matrix

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def check_dataset(dataset):
    dir = osp.join("/project2/depablo/erschultz", dataset, "samples")
    # dir = osp.join("/home/erschultz", dataset, "samples")
    ids = set()
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            file_dir = osp.join(dir, file)
            y_file = osp.join(file_dir, 'production_out', 'contacts600000.txt')
            if not osp.exists(y_file):
                ids.add(id)
            # try:
            #     x, psi, chi, chi_diag, e, s, y, ydiag = load_all(file_dir)
            #
            #     m, k = psi.shape
            #     seq = np.zeros((m, k))
            #     for i in range(k):
            #         seq_i = np.loadtxt(osp.join(file_dir, f'seq{i}.txt'))
            #         seq[:, i] = seq_i
            #
            #     if not np.array_equal(seq, psi):
            #         print(psi)
            #         print(seq)
            #         print(id)
            #         ids.add(id)
            # except Exception as e:
            #     print(f'id={id}: {e}')
            #     ids.add(id)
            #     continue

    print(ids, len(ids))

def check_dataset_p_s(dataset):
    dir = osp.join("/project2/depablo/erschultz", dataset, "samples")
    # dir = osp.join("/home/erschultz", dataset, "samples")
    ids = set()
    vals = np.zeros(10000)
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            file_dir = osp.join(dir, file)
            try:
                y, _ = load_Y(file_dir)
                y /= np.mean(y.diagonal())
                meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
                # if meanDist[10] > 0.06:
                #     ids.add(id)
                vals[id] = meanDist[10]

            except Exception as e:
                print(f'id={id}: {e}')
                ids.add(id)
                continue

    np.savetxt(osp.join(dir, 'vals.txt'), vals)
    np.savetxt(osp.join(dir, 'ids.txt'), ids)
    print(vals, len(vals))
    print(ids, len(ids))

def test_robust_PCA():
    if False:
        dir = '/home/eric/dataset_test/rpca_test'
        n = 1000
        r = 10
        x = np.random.rand(n, r)
        y = np.random.rand(r, n)
        l_0 = x @ y
        s_0 = np.random.binomial(n = 1, p = 0.3, size = (n, n))

        inp = l_0 + s_0

        plotContactMap(l_0, ofile = osp.join(dir, 'L.png'), vmax = 'max')
        plotContactMap(s_0, ofile = osp.join(dir, 'S.png'), vmax = 1)
        plotContactMap(inp, ofile = osp.join(dir, 'inp.png'), vmax = 'max')

        L, S = R_pca(inp).fit(max_iter=200)
        plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = np.mean(y))
        plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmax = np.mean(y))

    if True:
        dir = '/home/eric/dataset_test/rpca_test2'
        dataset_test = '/home/eric/dataset_test/samples'
        l0 = np.load(osp.join(dataset_test, 'sample20/PCA_analysis/y_diag_rank_1.npy'))
        p = np.load(osp.join(dataset_test, 'sample22/y.npy'))
        p = p / np.max(p) + 1e-8
        # m = np.load(osp.join(dataset_test, 'sample21/y.npy'))
        m = l0*p
        s0 = m - l0
        l0_log = np.log(l0)
        p_log = np.log(p)
        print('p', np.min(p_log))
        m_log = np.log(m)
        print(np.min(m_log))
        # plotContactMap(l0, ofile = osp.join(dir, 'L0.png'), vmax = 'max')
        # plotContactMap(l0_log, ofile = osp.join(dir, 'L0_log.png'), vmax = 'max')
        # plotContactMap(m, ofile = osp.join(dir, 'M.png'), vmax = 'mean')
        # plotContactMap(m_log, ofile = osp.join(dir, 'M_log.png'), vmin = 'min',
        #                 vmax = 'max')
        # plotContactMap(s0, ofile = osp.join(dir, 'S0.png'), vmin = 'min',
        #                 vmax = 'mean', cmap='blue-red')

        # plotContactMap(p, ofile = osp.join(dir, 'P.png'), vmax = 'mean')
        # plotContactMap(p_log, ofile = osp.join(dir, 'P_log.png'), vmin = 'min',
        #                 vmax = 'max')

        # L, S = R_pca(m).fit(max_iter=200)
        # plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = 'mean')
        # plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmin = 'min',
        #                 vmax = 'max', cmap='blue-red')
        L_log, S_log = R_pca(m_log).fit(max_iter=2000)
        plotContactMap(L_log, ofile = osp.join(dir, 'RPCA_L_log.png'), vmin = 'min',
                    vmax = 'max')
        plotContactMap(S_log, ofile = osp.join(dir, 'RPCA_S_log.png'), vmin = 'min',
                    vmax = 'max', cmap='blue-red')
        L_log_exp = np.exp(L_log)
        S_log_exp = np.exp(S_log)
        plotContactMap(L_log_exp, ofile = osp.join(dir, 'RPCA_L_log_exp.png'),
                    vmax = 'max')
        plotContactMap(S_log_exp, ofile = osp.join(dir, 'RPCA_S_log_exp.png'),
                    vmin = 'min', vmax = 'max', cmap='blue-red')

        PC_m = plot_top_PCs(m, inp_type='m', verbose = True, odir = dir, plot = True)
        meanDist = genomic_distance_statistics(m)
        m_diag = diagonal_preprocessing(m, meanDist)
        PC_m_diag = plot_top_PCs(m_diag, inp_type='m_diag', verbose = True,
                                odir = dir, plot = True)

        # plot_top_PCs(L_log_exp, inp_type='L_log_exp', verbose = True, odir = dir, plot = True)
        PC_L_log = plot_top_PCs(L_log, inp_type='L_log', verbose = True,
                                odir = dir, plot = True)
        stat = pearson_round(PC_L_log[0], PC_m[0])
        print("Correlation between PC 1 of L_log and M: ", stat)
        stat = pearson_round(PC_L_log[1], PC_m[1])
        print("Correlation between PC 2 of L_log and M: ", stat)
        meanDist = genomic_distance_statistics(L_log)
        L_log_diag = diagonal_preprocessing(L_log, meanDist)
        PC_L_log_diag = plot_top_PCs(L_log_diag, inp_type='L_log_diag',
                                    verbose = True, odir = dir, plot = True)
        stat = pearson_round(PC_L_log_diag[0], PC_m_diag[0])
        print("Correlation between PC 1 of L_log_diag and M_diag: ", stat)
        stat = pearson_round(PC_L_log_diag[1], PC_m_diag[1])
        print("Correlation between PC 2 of L_log_diag and M_diag: ", stat)
        # plot_top_PCs(m_log, inp_type='m_log', verbose = True, odir = dir, plot = True)
        # meanDist = genomic_distance_statistics(m_log)
        # m_log_diag = diagonal_preprocessing(m_log, meanDist)
        # plot_top_PCs(m_log_diag, inp_type='m_log_diag', verbose = True, odir = dir,
        #             plot = True)

    if False:
        # dir = '/home/eric/dataset_test/samples/sample104'
        dir = '/home/eric/sequences_to_contact_maps/dataset_09_21_21/samples/sample1'
        y = np.load(osp.join(dir, 'y.npy'))
        # L, S = R_pca(y).fit(max_iter=200)def time_comparison():
        # plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = np.mean(y))
        # plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmax = np.mean(y))

        # l = 1/np.sqrt(1024) * 1/100
        y_diag = np.load(osp.join(dir, 'y_diag.npy'))
        L, S = R_pca(y_diag).fit(max_iter=200)
        plotContactMap(L, ofile = osp.join(dir, 'RPCA_L_diag.png'), vmin='min',
                        vmax = 'max')
        plotContactMap(S, ofile = osp.join(dir, 'RPCA_S_diag.png'), vmin='min',
                        vmax = np.max(S), cmap='blue-red')
        plotContactMap(y_diag, ofile = osp.join(dir, 'y_diag.png'), vmax = 'max')
        plot_top_PCs(L, verbose = True)

def time_comparison():
    # dir = '/project2/depablo/erschultz/dataset_05_18_22/samples'
    dir = '/home/erschultz/sequences_to_contact_maps/dataset_05_18_22/samples'
    samples_per_size = 3
    num_sizes = 3

    times_dict = defaultdict(lambda: np.full([num_sizes, samples_per_size], np.nan))
    # dictionary with keys = method : vals = array of times with rows = sizes,
    # cols = replicate samples
    it = 1
    for row in range(num_sizes):
        for col in range(samples_per_size):
            print(it)
            sample_dir = osp.join(dir, f'sample{it}')
            max_ent_file = osp.join(sample_dir, 'max_ent_table_1.txt')
            it += 1
            if not osp.exists(max_ent_file):
                continue

            with open(osp.join(max_ent_file, ), 'r') as f:
                line = f.readline()
                while not line.startswith('Method'):
                    line = f.readline()
                line = f.readline()

                while not line.startswith('\end'):
                    if line.startswith('\hline'):
                        line = f.readline()
                        continue

                    line_list = line.split(' & ')
                    try:
                        if line_list[1] != '':
                            k = line_list[1]
                        method = line_list[0].replace('-normalize', '')
                        if 'diagMLP' in method:
                            method = '-'.join(method.split('-')[:-1])
                        if method == 'Ground Truth-diag_chi-E':
                            method = 'Ground Truth'
                        if 'GNN' in method:
                            method = 'GNN-' + '-'.join(method.split('-')[3:])
                        if k.isdigit():
                            label = f'{method}_k{k}'
                        else:
                            label = f'{method}'
                        print(label)
                        time = float(line_list[-1].split(' ')[0])
                    except:
                        print(line)
                        raise
                    times_dict[label][row,col] = time
                    line = f.readline()


    cmap = matplotlib.cm.get_cmap('tab20')
    fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    sizes = np.array([512., 1024., 2048., 4096.])[:num_sizes]
    sizes_shift = np.copy(sizes)
    ind = 0
    for method in sorted(times_dict.keys()):
        if method in {'GNN-'} or method.endswith('k2') or method.endswith('k4'):
            continue
        arr = times_dict[method][:num_sizes, :]
        print('method: ', method)
        print(arr, arr.shape)
        times = np.nanmean(arr, axis = 1)
        prcnt_converged = np.count_nonzero(~np.isnan(arr), axis = 1) / arr.shape[1]
        prcnt_converged[prcnt_converged == 0] = np.nan
        times_std = np.nanstd(arr, axis = 1)
        print('times', times)
        print('prcnt_converged', prcnt_converged)
        print('times std', times_std)
        print()

        # ax2.plot(sizes_shift, prcnt_converged, ls = '--', color = cmap(ind % cmap.N))
        ax.errorbar(sizes_shift, times, yerr = times_std, label = method,
                    color = cmap(ind % cmap.N), fmt = "o")
        sizes_shift += 20
        ind += 1


    ax.set_ylabel('Total Time (mins)', fontsize=16)
    # ax2.set_ylabel(f'Percent Converged (of {arr.shape[1]})', fontsize=16)
    ax.set_xlabel('Simulation size', fontsize=16)
    ax.set_ylim((0, None))
    ax.set_xticks(sizes) # , 2048, 4096
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'time.png'))
    plt.close()

def plot_sc_p_s():
    # quick function to plot sc p(s) curves
    dir = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017'
    data_dir = osp.join(dir, 'contact_diffusion_kNN8scc/iteration_1/sc_contacts')
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ifile = osp.join(dir, 'contact_diffusion_kNN8scc/iteration_0/sc_contacts/ifile_dict.json')
    with open(ifile) as f:
        ifile_dict = json.load(f)

    with open(osp.join(dir, 'samples/phase_dict.json'), 'r') as f:
        phase_dict = json.load(f)

    phase_count_dict = defaultdict(int) # how many times have we seen each phase
    phase_meanDist_dict = defaultdict(list) # maps phase to meandist
    for file in os.listdir(data_dir):
        if file.endswith('cool'):
            print(file)
            ifile = osp.join(data_dir, file)
            i = file.split('.')[0].split('_')[-1]

            dir = ifile_dict[osp.split(file)[1].replace('.cool', '.mcool')]
            phase = phase_dict[dir]
            phase_count_dict[phase] += 1

            y = load_contact_map(ifile, chrom=10, resolution=50000)

            ofile = osp.join(data_dir, 'sc_contacts_time', f'y_sc_{i}_chrom10.png')
            contacts = int(np.sum(y) / 2)
            sparsity = np.round(np.count_nonzero(y) / len(y)**2 * 100, 2)
            title = f'Sample {i}:\n# contacts: {contacts}, sparsity: {sparsity}%'
            plot_matrix(y, ofile, title, vmax = 'mean')


            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            meanDist[50:] = uniform_filter(meanDist[50:], 3, mode = 'constant')
            phase_meanDist_dict[phase].append(meanDist[:1024])

    for phase in ['G1', 'S', 'G2', 'pre-M', 'post-M']:
        print(f'{phase}: {phase_count_dict[phase]} samples')
        if phase_count_dict[phase] > 0:
            meanDist_list = phase_meanDist_dict[phase]
            meanDist_arr = np.array(meanDist_list)
            meanDist = np.mean(meanDist_arr, 0)
            meanDist_stdev = np.std(meanDist_arr, 0, ddof=1)
            x = np.arange(0, len(meanDist))
            ax.plot(x, meanDist, label = f'{phase} (mean)')
            ax.fill_between(x, meanDist - meanDist_stdev,
                            meanDist + meanDist_stdev, alpha = 0.5)
    #
    # y = np.load('/home/erschultz/sequences_to_contact_maps/single_cell_nagano_imputed/' /
                # 'samples/sample390/none-diagMLP-66/k0/replicate1/y.npy')
    # meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    # ax.plot(meanDist, label = 'MLP-66', color = 'k')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    ax.legend()
    ax.set_ylim(10**-5, None)
    plt.tight_layout()
    plt.savefig(osp.join(data_dir, 'sc_contacts_time', 'meanDist_log2.png'))
    plt.close()

def plot_mean_dist_S(dataset, experimental=False, label=None):
    # plot different p(s) curves
    dir = '/home/erschultz/'
    data_dir = osp.join(dir, dataset)

    data = defaultdict(dict) # sample : {meanDist, diag_chis_step} : vals
    samples, experimental = get_samples(dataset)
    for sample in samples:
        sample_dir = osp.join(data_dir, 'samples', f'sample{sample}')
        if experimental:
            max_ent_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03-max_ent')
            L = load_max_ent_L(max_ent_dir)
            D = load_max_ent_D(max_ent_dir)
            S = calculate_S(L, D)
        else:
            S = np.load(osp.join(sample_dir, 'S.npy'))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(S, 'freq')
        data[sample]['meanDist'] = meanDist

    fig, ax = plt.subplots()
    for i, sample in enumerate(data.keys()):
        meanDist = data[sample]['meanDist']
        X = np.arange(0, len(meanDist), 1)
        if i > 10:
            ls = 'dashed'
        else:
            ls = 'solid'
        if label is not None:
            ax.plot(X, meanDist, label = data[sample][label], ls = ls)
        else:
            ax.plot(X, meanDist, label = sample, ls = ls)

    ax.set_xscale('log')
    ax.set_ylabel('Mean Energy', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)

    if label is not None:
        ax.legend(loc='upper right', title = label)
    else:
        ax.legend(loc='upper right', title = 'Sample')
    plt.tight_layout()
    plt.savefig(osp.join(data_dir, 'S_meanDist.png'))
    plt.close()

def time_comparison_dmatrix():
    dir = '/home/erschultz/dataset_test/samples'

    time_dict_e = defaultdict(list) # m : list of times (when use_e = True)
    time_dict = defaultdict(list) # m : list of times
    for sample in range(1, 25):
        sample_dir = osp.join(dir, f'sample{sample}')
        log_file = osp.join(sample_dir, 'log.log')

        if osp.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    if line.startswith('Took'):
                        time = int(line.split(' ')[1][:-7])
                        time /= 60
                    if line.startswith('contactmap size'):
                        m = int(line.split(': ')[1])

        config_file = osp.join(sample_dir, 'config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        if config['ematrix_on']:
            time_dict_e[m].append(time)
        else:
            time_dict[m].append(time)

    print(time_dict_e)
    print(time_dict)

    m_arr = np.array([512, 1024, 2048, 4096], dtype = np.float64)
    time_mean = np.zeros_like(m_arr)
    time_std = np.zeros_like(m_arr)
    time_e_mean = np.zeros_like(m_arr)
    time_e_std = np.zeros_like(m_arr)
    for i, m in enumerate(m_arr):
        time = time_dict[m]
        time_mean[i] = np.mean(time)
        time_std[i] = np.std(time, ddof = 1)

        time_e = np.array(time_dict_e[m])
        time_e_mean[i] = np.mean(time_e)
        time_e_std[i] = np.std(time_e, ddof = 1)

    print('m', time_e_mean, 'std', time_e_std)
    prcnt_diff = time_e_mean / time_mean * 100
    print(prcnt_diff)


    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.scatter(m_arr, time_mean, color = 'blue', label = 'Original')
    ax.errorbar(m_arr, time_mean, yerr = time_std, ls='none', color = 'blue')
    ax.scatter(m_arr, time_e_mean, color = 'red', label = 'Net Energy')
    ax.errorbar(m_arr, time_e_mean, yerr = time_e_std, ls='none', color = 'red')
    ax2.plot(m_arr, prcnt_diff, color = 'black', label='% of original')

    ax.set_ylabel('Total Time (mins)', fontsize=16)
    ax2.set_ylabel('% of Original', fontsize=16)
    ax.set_xlabel('Simulation size', fontsize=16)
    ax.set_ylim((0, 84))
    ax2.set_ylim((None, 74))
    ax.set_xticks(m_arr)
    ax.legend(loc='upper left', title='Total Time')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'time.png'))
    plt.close()


    time_dict_e = defaultdict(list) # num_diag_chis : list of times (when use_e = True)
    time_dict = defaultdict(list) # num_diag_chis : list of times
    for sample in range(100, 106):
        sample_dir = osp.join(dir, f'sample{sample}')

        log_file = osp.join(sample_dir, 'log.log')
        if osp.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    if line.startswith('Took'):
                        time = int(line.split(' ')[1][:-7])

        config_file = osp.join(sample_dir, 'config.json')
        if osp.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
                k = len(config['diag_chis'])


        if osp.exists(osp.join(sample_dir, 'd_matrix.txt')):
            time_dict_e[k].append(time)
        else:
            time_dict[k].append(time)

    print(time_dict)
    print(time_dict_e)

    k_arr = np.array([32, 64, 128])
    time_mean = np.zeros_like(k_arr)
    time_std = np.zeros_like(k_arr)
    time_e_mean = np.zeros_like(k_arr)
    time_e_std = np.zeros_like(k_arr)
    for i, k in enumerate(k_arr):
        time = time_dict[k]
        time_mean[i] = np.mean(time)
        # time_std[i] = np.std(time, ddof = 1)

        time_e = time_dict_e[k]
        time_e_mean[i] = np.mean(time_e)
        # time_e_std[i] = np.std(time_e, ddof = 1)

    fig, ax = plt.subplots()
    ax.scatter(k_arr, time_mean, color = 'blue', label = 'Original')
    ax.scatter(k_arr, time_e_mean, color = 'red', label = 'Net Energy')

    ax.set_ylabel('Total Time (seconds)', fontsize=16)
    ax.set_xlabel('Number of Diagonal Bins', fontsize=16)
    ax.set_ylim((0, None))
    ax.set_xticks(k_arr)
    ax.legend(loc='upper left', title='Total Time')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'time2.png'))
    plt.close()

def max_ent_loss_for_gnn(dataset, sample):
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}'

    PCA_dir = osp.join(dir, 'PCA_split-binarizeMean-E/k12/replicate1')
    PCA_dir_final = get_final_max_ent_folder(PCA_dir)
    os.chdir(PCA_dir_final)
    sim = Sim("production_out")
    sim.plot_obs_vs_goal('test.png')

    GNN_dir = osp.join(dir, 'GNN-289-E/k0/replicate1')
    y = np.load(osp.join(GNN_dir, 'y.npy'))
    y = y.astype(float) # ensure float
    y /= np.mean(np.diagonal(y))
    sim.verbose = False
    print(sim.seqs)
    plaid = get_plaid_goal(y, sim, sim.seqs)
    diag = get_diag_goal(y, sim)
    sim.obs_tot = np.hstack((plaid, diag))
    sim.plot_obs_vs_goal('test2.png')

def compare_scc_bio_replicates():
    dir = '/home/erschultz/dataset_test/samples'
    y_a = np.load(osp.join(dir, 'sample2201/y.npy'))
    y_b = np.load(osp.join(dir, 'sample2202/y.npy'))
    scc = SCC()
    corr_scc = scc.scc(y_a, y_b, var_stabilized = False)
    corr_scc_var = scc.scc(y_a, y_b, var_stabilized = True)
    print(corr_scc, corr_scc_var)


    dir = '/home/erschultz/dataset_test/samples'
    y_a = np.load(osp.join(dir, 'sample2201/y.npy'))
    y_b = np.load(osp.join(dir, 'sample2204/y.npy'))
    scc = SCC()
    corr_scc = scc.scc(y_a, y_b, var_stabilized = False)
    corr_scc_var = scc.scc(y_a, y_b, var_stabilized = True)
    print(corr_scc, corr_scc_var)

    dir = '/home/erschultz/dataset_test/samples'
    y_a = np.load(osp.join(dir, 'sample2202/y.npy'))
    y_b = np.load(osp.join(dir, 'sample2204/y.npy'))
    scc = SCC()
    corr_scc = scc.scc(y_a, y_b, var_stabilized = False)
    corr_scc_var = scc.scc(y_a, y_b, var_stabilized = True)
    print(corr_scc, corr_scc_var)


    dir = '/home/erschultz/dataset_test/samples'
    y_a = np.load(osp.join(dir, 'sample2203/y.npy'))
    y_b = np.load(osp.join(dir, 'sample2204/y.npy'))
    scc = SCC()
    corr_scc = scc.scc(y_a, y_b, var_stabilized = False)
    corr_scc_var = scc.scc(y_a, y_b, var_stabilized = True)
    print(corr_scc, corr_scc_var)

def gnn_of_max_ent(samples, k, ID):
    '''Analysis of results for running GNN on max ent of experimental data'''
    dir = '/home/erschultz/dataset_02_04_23/samples'
    scc = SCC()
    for s in samples:
        s_dir = osp.join(dir, f'sample{s}')
        y_exp = np.load(osp.join(s_dir, 'y.npy'))
        y_gnn1 = np.load(osp.join(s_dir, f'GNN-{ID}-E/k0/replicate1/y.npy'))

        s_dir = osp.join(s_dir, f'PCA-normalize-E/k{k}/replicate1',
                        f'samples/sample{s}_copy')
        assert osp.exists(s_dir), s_dir
        y_max_ent1 = np.load(osp.join(s_dir, 'y.npy'))

        max_ent_dir = osp.join(s_dir, 'PCA-normalize-E/k8/replicate1')
        y_max_ent2 = np.load(osp.join(max_ent_dir, 'y.npy'))

        gnn_dir = osp.join(s_dir, f'GNN-{ID}-S/k0/replicate1')
        y_gnn2 = np.load(osp.join(gnn_dir, 'y.npy'))

        print('SCC:')
        corr = scc.scc(y_exp, y_gnn1, var_stabilized = True)
        print('Exp vs GNN1:', np.round(corr, 3))
        corr = scc.scc(y_exp, y_gnn2, var_stabilized = True)
        print('Exp vs GNN2:', np.round(corr, 3))

        corr = scc.scc(y_exp, y_max_ent1, var_stabilized = True)
        print('Exp vs Max Ent 1:', np.round(corr, 3))
        corr = scc.scc(y_exp, y_max_ent2, var_stabilized = True)
        print('Exp vs Max Ent 2:', np.round(corr, 3))

        corr = scc.scc(y_max_ent1, y_max_ent2, var_stabilized = True)
        print('Max Ent 1 vs Max Ent 2:', np.round(corr, 3))
        corr = scc.scc(y_max_ent1, y_gnn2, var_stabilized = True)
        print('Max Ent 1 vs GNN2:', np.round(corr, 3))

def make_dataset_of_converged(dataset):
    '''
    Inputs:
        dataset: filename of experimental dataset
    '''
    dir = '/project2/depablo/erschultz'
    # dir = '/home/erschultz'
    data_dir = osp.join(dir, dataset)
    odata_dir = osp.join(dir, f'{dataset}-max_ent')
    s_dir = osp.join(data_dir, 'samples')
    os_dir = osp.join(odata_dir, 'samples')

    if not osp.exists(odata_dir):
        os.mkdir(odata_dir, mode=0o755)
        os.mkdir(os_dir, mode=0o755)

    converged_count = 0
    count = 0
    for file in os.listdir(s_dir):
        if file.startswith('sample'):
            f_dir = osp.join(s_dir, file)
            pca_dir = osp.join(f_dir, 'PCA-normalize-S/k8/replicate1')
            conv_file = osp.join(pca_dir, 'convergence.txt')
            converged = False
            eps = 1e-3
            if osp.exists(conv_file):
                conv = np.loadtxt(conv_file)
                for ind in range(1, len(conv)):
                    diff = conv[ind] - conv[ind-1]
                    if np.abs(diff) < eps and conv[ind] < conv[0]:
                        converged = True

                if converged:
                    final_it_dir = get_final_max_ent_folder(pca_dir)
                    converged_count += 1
                    of_dir = osp.join(os_dir, file)
                    if not osp.exists(of_dir):
                        os.mkdir(of_dir)

                    shutil.copy(osp.join(pca_dir, 'y.npy'), osp.join(of_dir, 'y.npy'))
                    shutil.copy(osp.join(pca_dir, 'L.npy'), osp.join(of_dir, 'L.npy'))
                    shutil.copy(osp.join(pca_dir, final_it_dir, 'config.json'), osp.join(of_dir, 'config.json'))
                    with open(osp.join(pca_dir, final_it_dir, 'config.json')) as f:
                        config = json.load(f)
                    diag_chi_continuous = calculate_diag_chi_step(config)
                    np.save(osp.join(of_dir, 'diag_chis_continuous.npy'), diag_chi_continuous)

                    # shutil.copytree(osp.join(final_it_dir, 'production_out'), osp.join(of_dir, 'data_out'))
                else:
                    count += 1
    print(f'{converged_count} out of {converged_count+count} converged')

def compare_y_exp_vs_sim():
    sim = '/home/erschultz/dataset_05_28_23/samples/sample324'
    exp = '/home/erschultz/dataset_04_10_23/samples/sample1002'
    GNN_ID=411
    model_path = f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{GNN_ID}'
    config = default.config
    root = "optimize_grid_b_140_phi_0.03"

    def print_arr(arr, name=None):
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().detach().numpy()
        if name is not None:
            print(name)
        print(arr, arr.shape, np.nanmin(arr), np.nanpercentile(arr, [1,99]), np.nanmax(arr))

    for dir in [sim, exp]:
        print(dir)
        y = np.load(osp.join(dir, 'y.npy')).astype(np.float64)
        y /= np.mean(np.diagonal(y))
        m = len(y)


        config['nbeads'] = len(y)
        config['grid_size'] = np.loadtxt(osp.join(dir, root, 'grid_size.txt'))
        getenergy = GetEnergy(config = config)

        model, data, dataset = getenergy.get_energy_gnn(model_path, dir,
                                    grid_path=osp.join(root, 'grid_size.txt'),
                                    verbose = False, return_model_data = True)

        print_arr(data.edge_attr[:, 1], 'i-j')
        print_arr(data.edge_attr[:, 0], 'H_ij')
        arr = data.edge_attr[:, 0] * 10
        bin_width=1
        # plt.hist(arr)
        # plt.hist(arr, label = osp.split(dir)[1], alpha = 0.5,
        #             weights = np.ones_like(arr) / len(arr),
        #             bins = range(math.floor(np.nanmin(arr)), math.ceil(np.nanmax(arr)) + bin_width, bin_width))
    #
    # plt.legend()
    # plt.show()

        # print_arr(hat_S)
        latent1, latent2 = model.latent(data)
        print_arr(latent1, 'latent1')
        print_arr(latent2, 'latent2')

        plaid1 = model.plaid_component(latent1)
        plaid2 = model.plaid_component(latent2)
        print_arr(plaid1, 'plaid1')
        print_arr(plaid2, 'plaid2')

        diag1 = model.diagonal_component(latent1).detach()[0, 0, :]
        diag1 = np.multiply(np.sign(diag1), np.exp(np.abs(diag1)) - 1)
        diag2 = model.diagonal_component(latent2).detach()[0, 0, :]
        diag2 = np.multiply(np.sign(diag2), np.exp(np.abs(diag2)) - 1)
        print_arr(diag1, 'diag1')
        print_arr(diag2, 'diag2')

        S1 = plaid1 + diag1
        S2 = plaid2 + diag2
        print_arr(S1, 'S1')
        print_arr(S2, 'S2')
        #
        # yhat = model(data)
        # yhat = yhat.cpu().detach().numpy().reshape((m, m))
        # print_arr(yhat, 'yhat')
        # yhat = np.multiply(np.sign(yhat), np.exp(np.abs(yhat)) - 1)
        # print_arr(yhat, 'yhat_trans')

def edit_setup(dataset, exp_dataset):
    dir = f'/home/erschultz/{dataset}/setup'
    grid_root = f'optimize_grid_b_140_phi_0.03'
    for i in range(1, 5001):
        file = osp.join(dir, f'sample_{i}.txt')
        rowList = []
        with open(file, 'r') as f:
            is_j = False
            j = -1
            for line in f:
                line = line.strip()
                rowList.append([line])
                if is_j:
                    j = int(line)
                if line.startswith('--exp_max_ent'):
                    is_j = True
                else:
                    is_j = False

        rowList.append(['--diag_chi_experiment'])
        rowList.append([osp.join(exp_dataset, f'samples/sample{j}', grid_root)])


        with open(file, 'w') as f:
            wr = csv.writer(f)
            wr.writerows(rowList)

def make_small(dataset):
    dir = f'/home/erschultz/{dataset}/samples'
    odir = f'/home/erschultz/{dataset}-small'
    grid_root = f'optimize_grid_b_180_phi_0.008_spheroid_1.5'
    if not osp.exists(odir):
        os.mkdir(odir)
    odir = osp.join(odir, 'samples')
    if not osp.exists(odir):
        os.mkdir(odir)
    for s in os.listdir(dir):
        s_dir_grid = osp.join(dir, s, grid_root)
        s_odir = osp.join(odir, s)
        if not osp.exists(s_odir):
            os.mkdir(s_odir)
        s_odir_grid = osp.join(s_odir, grid_root)
        if not osp.exists(s_odir_grid):
            os.mkdir(s_odir_grid)
        for f in ['y.npy', 'grid.txt']:
            shutil.copyfile(osp.join(s_dir_grid, f), osp.join(s_odir_grid, f))

def test_convergence(dataset, mode='loss'):
    dir = f'/home/erschultz/{dataset}/samples'
    b=180; phi=0.008; ar=1.5
    samples, _ = get_samples(dataset, test=True)
    samples = samples[:10]
    cmap = matplotlib.cm.get_cmap('tab10')
    fig, ax = plt.subplots()

    for ls, max_ent_mode in zip(['-'], ['_700k']):
        if max_ent_mode is None:
            continue
        for i, s in enumerate(samples):
            s_dir = osp.join(dir, f'sample{s}')
            fpath = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}_spheroid_{ar}-max_ent5{max_ent_mode}')
            assert osp.exists(fpath), fpath

            if mode == 'loss':
                conv = np.loadtxt(osp.join(fpath, 'convergence.txt'))
                max_it = len(conv)

                convergence = []
                eps = 1e-2
                converged_it = None
                for j in range(1, max_it):
                    diff = conv[j] - conv[j-1]
                    convergence.append(np.abs(diff))
            elif mode.startswith('param'):
                all_chis = []
                all_diag_chis = []
                for j in range(31):
                    it_path = osp.join(fpath, f'iteration{j}')
                    if osp.exists(it_path):
                        config_file = osp.join(it_path, 'production_out/config.json')
                        with open(config_file) as f:
                            config = json.load(f)
                        chis = np.array(config['chis'])
                        chis = chis[np.triu_indices(len(chis))] # grab upper triangle
                        diag_chis = np.array(config['diag_chis'])

                        all_chis.append(chis)
                        all_diag_chis.append(diag_chis)

                params = np.concatenate((all_diag_chis, all_chis), axis = 1)

                if mode == 'param':
                    convergence = []
                    for j in range(1, len(params)):
                        diff = params[j] - params[j-1]
                        conv = np.linalg.norm(diff, ord = 2)
                        convergence.append(conv)
                elif mode == 'param_mag':
                    convergence = []
                    for j in range(1, len(params)):
                        diff = np.linalg.norm(params[j]) - np.linalg.norm(params[j-1])
                        conv = np.abs(diff)
                        convergence.append(conv)
            ax.plot(convergence, ls = ls, c = cmap(i % cmap.N))


    ax2 = ax.twinx()
    for i, s in enumerate(samples):
        ax2.plot(np.NaN, np.NaN, c = cmap(i % cmap.N), label = f'sample {s}')
    ax2.get_yaxis().set_visible(False)
    if len(samples) < 20:
        ax2.legend(title='sample')

    ax.set_yscale('log')
    if mode == 'param':
        ax.set_ylabel(r'$|\chi_{i}-\chi_{i-1}|$ (L2 norm)', fontsize=16)
        # ax.axhline(100, c='k', ls='--')
        ax.axhline(10, c='k', ls='--')
        # ax.axhline(1, c='k', ls='--')
    if mode == 'param_mag':
        ax.set_ylabel(r'abs$(|\chi_{i}|-|\chi_{i-1}|)$', fontsize=16)
        ax.axhline(1, c='k', ls='--')
        ax.axhline(1e-1, c='k', ls='--')
    elif mode == 'loss':
        ax.set_ylabel(r'$|f(\chi)_{i}-f(\chi)_{i-1}|$', fontsize=16)
        ax.axhline(1e-2, c='k', ls='--')
        ax.axhline(1e-3, c='k', ls='--')
    ax.set_xlabel('Iteration', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'/home/erschultz/TICG-chromatin/figures/conv_{mode}.png')
    plt.close()

def compare_s_per_iteration():
    dir = '/home/erschultz/Su2020/samples/sample1013'
    max_ent_dir = osp.join(dir, 'optimize_grid_b_261_phi_0.01-max_ent10_longer')
    assert osp.exists(max_ent_dir)
    S_10 = np.load(osp.join(max_ent_dir, 'iteration10/S.npy'))
    S_20 = np.load(osp.join(max_ent_dir, 'iteration20/S.npy'))
    diff = S_20 - S_10

    fig, (ax1, ax2, ax3, axcb) = plt.subplots(1, 4,
                                gridspec_kw={'width_ratios':[1,1,1,0.08]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    arr = np.array([S_10, S_20])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    s1 = sns.heatmap(S_10, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax1, cbar = False)
    s1.set_title(f'$S$ iteration 10', fontsize = 16)
    s1.set_yticks([])
    s2 = sns.heatmap(S_20, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax2, cbar = False)
    s2.set_title(f'$S$ iteration 20', fontsize = 16)
    s2.set_yticks([])
    s3 = sns.heatmap(diff, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax3, cbar_ax = axcb)
    title = ('Difference\n'
            '20 - 10')
    s3.set_title(title, fontsize = 16)
    s3.set_yticks([])
    plt.show()

def compare_p_s():
    dir = '/home/erschultz'
    datasets = ['Su2020', 'dataset_02_04_23']
    samples = [1013, 243]
    cell_lines = ['IMR90', 'GM12878']

    for dataset, sample, cell_line in zip(datasets, samples, cell_lines):
        s_dir = osp.join(dir, dataset, f'samples/sample{sample}')
        y = np.load(osp.join(s_dir, 'y.npy')).astype(float)
        y /= np.mean(np.diagonal(y))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
        plt.plot(meanDist, label = f'{cell_line} sample {sample}')

    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Beads', fontsize=16)
    plt.legend()
    plt.show()

def test_pooling():
    dir = '/home/erschultz/timing_analysis'
    y_256 = np.load(osp.join(dir, '256', 'samples/sample1/y.npy'))
    for i in [512, 1024]:
        y_i = np.load(osp.join(dir, str(i), 'samples/sample1/y.npy'))
        y_rescale = rescale_matrix(y_i, int(i/256))
        print(y_rescale.shape)
        print(np.allclose(y_256, y_rescale))


def test_harmonic_angle():
    # p1: 314.257 314.257 314.257
    # p2: 482.553 150.319 200.583
    # p3: 502.703 387.782 307.008
    # disp1: -168.296  163.938  113.674
    # disp2: 20.1498 237.463 106.425
    # theta: 0.796404
    bead1 = np.array([314.257, 314.257, 314.257]).astype(float)
    bead2 = np.array([482.553, 150.319, 200.583]).astype(float)
    bead3 = np.array([502.703, 387.782, 307.008]).astype(float)
    a = bead2 - bead1
    b = bead2 - bead3
    print(a, b)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    print(a, b)
    cos_theta = np.dot(a,b)
    print(cos_theta)
    theta = np.arccos(cos_theta)
    print(theta)

def fene(r, k, R0):
    if r < R0:
        return -0.5*k*(R0**2)*math.log(1-r/R0)
    return 0

def hc(r, b):
    if r < b*(2**(1/6)):
        return 4*((b/r)**12-(b/r)**6+1/4)
    return 0

def test_FENE():
    b = 261
    k = 30 / b**2
    R0 = 1.5 * b
    X = np.arange(0.7*b, 1.6*b, b/100)
    Y1 = np.zeros_like(X)
    Y2 = np.zeros_like(X)
    Y3 = np.zeros_like(X)
    for i, x in enumerate(X):
        val1 = fene(x, k, R0)
        Y1[i] = val1
        val2 = hc(x, b)
        Y2[i] = val2
        Y3[i] = val1 + val2

    plt.plot(X, Y1, label='fene')
    plt.plot(X, Y2, label='hc')
    plt.plot(X, Y3, label='combined')
    plt.legend()
    plt.show()

def convergence():
    dir = '/home/erschultz/dataset_02_04_23/samples/sample209/optimize_grid_b_180_phi_0.008_spheroid_1.5-max_ent5_longer'
    conv = np.loadtxt(osp.join(dir, 'convergence.txt'))
    plt.plot(conv)
    plt.yscale('log')
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Iteration', fontsize=16)
    plt.savefig(osp.join(dir, 'convergence.png'))
    plt.close()

    diff_list = []
    eps = 1e-3
    for j in range(1, len(conv)):
        diff = conv[j] - conv[j-1]
        diff = np.abs(diff)
        diff_list.append(diff)
        if diff < eps and conv[j] < conv[0]:
            converged_it = j
            break

    print(diff_list)
    plt.plot(range(1, len(diff_list)+1), diff_list)
    plt.yscale('log')
    plt.axhline(1e-2, c='k', ls='--')
    plt.axhline(1e-3, c='k', ls='--')
    plt.ylabel(r'|Loss$_i$ - Loss$_{i-1}$|', fontsize=16)
    plt.xlabel('Iteration', fontsize=16)
    plt.savefig(osp.join(dir, 'convergence_diff.png'))
    plt.close()

def data_t_test():
    dataset = 'dataset_02_04_23'
    samples, _ = get_samples(dataset, test = True)
    samples_list = samples[:10]
    args = getArgs(data_folder = f'/home/erschultz/{dataset}',
                    samples = samples_list)
    args.experimental = True
    args.verbose = False
    args.bad_methods = ['_stop', 'b_140', 'b_261', 'spheroid_2.0', 'max_ent10']
    args.convergence_definition = 'normal'
    args.gnn_id = []
    data, _ = load_data(args)

    k = 5
    grid_root = 'optimize_grid_b_180_phi_0.008_spheroid_1.5'
    max_ent_dir = f'{grid_root}-max_ent{k}'
    metric = 'scc_var'
    method_1 = max_ent_dir
    result1 = np.array(data[k][method_1][metric], dtype=np.float64)
    print(result1, np.mean(result1))

    args.convergence_definition = 'strict'
    data, _ = load_data(args)
    method_2 = max_ent_dir + '_700k'
    result2 = np.array(data[k][method_2][metric], dtype=np.float64)
    print(result2, np.mean(result2))


    stat, pval = ss.ttest_rel(result2, result1)
    mean_effect_size = np.mean(result2 - result1)
    mean_effect_size = np.round(mean_effect_size, 3)
    print(stat, pval)
    print(mean_effect_size)

if __name__ == '__main__':
    # test_robust_PCA()
    # test_pooling()
    # test_convergence('dataset_02_04_23', 'loss')
    # test_convergence('dataset_02_04_23', 'param_mag')
    # check_dataset('dataset_09_25_22')
    check_dataset_p_s('dataset_09_28_23')
    # time_comparison()
    # time_comparison_dmatrix()
    # main()
    # compare_scc_bio_replicates()
    # max_ent_loss_for_gnn('dataset_11_14_22', 2201)
    # plot_mean_dist_S('dataset_09_17_23')
    # gnn_of_max_ent([207], 8, 378)
    # check_interpolation()
    # make_dataset_of_converged('dataset_03_21_23')
    # check_bonded_distributions()
    # compare_y_exp_vs_sim()
    # edit_setup('dataset_05_28_23', 'dataset_04_10_23')
    # edit_setup('dataset_04_28_23', 'dataset_02_04_23')
    # edit_setup('dataset_05_15_23', 'dataset_02_04_23')
    # make_small('dataset_02_04_23')
    # convergence()
    # data_t_test()
    # compare_s_per_iteration()
    # compare_p_s()
    # test_harmonic_angle()
    # test_FENE()
