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
import scipy.stats as ss
import seaborn as sns
import torch
import torch_geometric
from modify_maxent import plaid_dist
from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

sys.path.append('/home/erschultz/TICG-chromatin/maxent/bin')
from analysis import Sim
from get_goal_experimental import get_diag_goal, get_plaid_goal

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import (finalize_opt,
                                                              get_base_parser)
from sequences_to_contact_maps.scripts.clean_directories import \
    clean_directories
from sequences_to_contact_maps.scripts.energy_utils import (
    calculate_D, calculate_diag_chi_step, calculate_E_S, calculate_S,
    calculate_SD_ED)
from sequences_to_contact_maps.scripts.knightRuiz import knightRuiz
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_all, load_contact_map, load_Y)
from sequences_to_contact_maps.scripts.neural_nets.utils import (
    get_dataset, load_saved_model)
from sequences_to_contact_maps.scripts.plotting_utils import (plot_matrix,
                                                              plot_seq_binary)
from sequences_to_contact_maps.scripts.R_pca import R_pca
from sequences_to_contact_maps.scripts.similarity_measures import SCC
from sequences_to_contact_maps.scripts.utils import (DiagonalPreprocessing,
                                                     pearson_round,
                                                     triu_to_full)


def check_dataset(dataset):
    dir = osp.join("/project2/depablo/erschultz", dataset, "samples")
    # dir = osp.join("/home/erschultz", dataset, "samples")
    ids = set()
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            file_dir = osp.join(dir, file)
            try:
                x, psi, chi, chi_diag, e, s, y, ydiag = load_all(file_dir)

                m, k = psi.shape
                seq = np.zeros((m, k))
                for i in range(k):
                    seq_i = np.loadtxt(osp.join(file_dir, f'seq{i}.txt'))
                    seq[:, i] = seq_i

                if not np.array_equal(seq, psi):
                    print(psi)
                    print(seq)
                    print(id)
                    ids.add(id)
            except Exception as e:
                print(f'id={id}: {e}')
                ids.add(id)
                continue

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

def convergence_check():
    dir = '/home/erschultz/dataset_09_30_22/samples'
    results_1 = {} # param convergence
    results_2 = {} # loss convergence
    for file in os.listdir(dir):
        sample = osp.join(dir, file)
        if file.startswith('sample') and osp.isdir(sample):
            id = int(file[6:])
            for file in os.listdir(sample):
                if file.startswith('GNN-') or file.startswith('PCA-'):
                    method = osp.join(sample, file)
                    if 'MLP' in method and 'GNN' in method:
                        continue
                    for file in os.listdir(method):
                        if file.startswith('k'):
                            k = file[1:]
                            k_folder = osp.join(method, file)
                            for file in os.listdir(k_folder):
                                rep = file[-1]
                                if file[-1] == '2':
                                    continue
                                replicate = osp.join(k_folder, file)
                                label = f'{id}_{osp.split(method)[1]}_k{k}_r{rep}'

                                chis_diag_file = osp.join(replicate, 'chis_diag.txt')
                                if osp.exists(chis_diag_file):
                                    params = np.loadtxt(chis_diag_file)

                                chis_file = osp.join(replicate, 'chis.txt')
                                if osp.exists(chis_file):
                                    chis = np.loadtxt(chis_file)
                                    params = np.concatenate((params, chis), axis = 1)
                                else:
                                    continue

                                vals = []
                                for i in range(2, len(params)):
                                    diff = params[i] - params[i-1]
                                    vals.append(np.linalg.norm(diff, ord = 2))
                                results_1[label] = vals


                                conv_file = osp.join(replicate, 'convergence.txt')
                                if osp.exists(conv_file):
                                    conv = np.loadtxt(conv_file)
                                else:
                                    continue

                                vals = []
                                for i in range(1, len(conv)):
                                    diff = conv[i] - conv[i-1]
                                    vals.append(np.abs(diff))
                                results_2[label] = vals


    cmap = matplotlib.cm.get_cmap('tab10')
    ls_arr = ['solid', 'dotted', 'dashed', 'dashdot']
    for label, vals in results_1.items():
        print(label)
        id = int(label.split('_')[0])
        k = int(label[-4])

        plt.plot(vals, label = label, ls = ls_arr[k // 2])

    eps = 1
    plt.axhline(eps, c = 'k')
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel(r'$||x_t - x_{t-1}||_2$', fontsize=16)
    plt.legend()
    plt.title(r'$||x_t - x_{t-1}||_2 < $'+f'{eps}')
    plt.savefig(osp.join(dir, 'param_convergence.png'))
    plt.close()


    cmap = matplotlib.cm.get_cmap('tab10')
    ls_arr = ['solid', 'dotted', 'dashed', 'dashdot']
    for label, vals in results_2.items():
        print(label)
        id = int(label.split('_')[0])
        k = int(label[-4])

        plt.plot(vals, label = label, ls = ls_arr[k // 2])

    eps = 1e-3
    plt.axhline(eps, c = 'k')
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel(r'$|f(x_t) - f(x_{t-1})|$', fontsize=16)
    plt.legend(loc = 'upper right')
    plt.title(r'$|f(x_t) - f(x_{t-1})| < 10^{-3}$')
    plt.savefig(osp.join(dir, 'loss_convergence.png'))
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

def plot_p_s(dataset, experimental=False, params=False):
    # plot different p(s) curves
    dir = '/home/erschultz/'
    if experimental:
        data_dir = osp.join(dir, 'dataset_11_14_22/samples/sample1') # experimental data sample
        file = osp.join(data_dir, 'y.npy')
        y_exp = np.load(file)
        meanDist_ref = DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob')

    data_dir = osp.join(dir, dataset)

    data = defaultdict(dict) # sample : {meanDist, diag_chis_step} : vals
    for sample in [3001, 3002, 3003]:
        sample_dir = osp.join(data_dir, 'samples', f'sample{sample}')
        ifile = osp.join(sample_dir, 'y.npy')
        if osp.exists(ifile):
            y = np.load(ifile)
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            data[sample]['meanDist'] = meanDist


            if params:
                with open(osp.join(sample_dir, 'config.json')) as f:
                    config = json.load(f)
                diag_chis_step = calculate_diag_chi_step(config)
                data[sample]['diag_chis_step'] = np.array(diag_chis_step)


    for norm in [True, False]:
        fig, ax = plt.subplots()
        if params:
            ax2 = ax.twinx()
        if experimental:
            if norm:
                X = np.arange(0, 1, len(meanDist_ref))
            else:
                X = np.arange(0, len(meanDist_ref), 1)
            ax.plot(meanDist_ref, label = 'Experiment', color = 'k')

        for sample in data.keys():
            label = sample
            meanDist = data[sample]['meanDist']
            if norm:
                X = np.linspace(0, 1, len(meanDist))
            else:
                X = np.arange(0, len(meanDist), 1)
            ax.plot(X, meanDist, label = label)

            if params:
                diag_chis_step = data[sample]['diag_chis_step']
                print(X.shape, diag_chis_step.shape)
                ax2.plot(X, diag_chis_step, ls = '--', label = 'Parameters')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Contact Probability', fontsize = 16)
        ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)

        if params:
            ax.legend(loc='upper left', title = 'Sample')
            ax2.set_xscale('log')
            ax2.set_ylabel('Diagonal Parameter', fontsize = 16)
            ax2.legend(loc='upper right')
        else:
            ax.legend(loc='upper right', title = 'Sample')
        plt.tight_layout()
        plt.savefig(osp.join(data_dir, f'meanDist_norm_{norm}.png'))
        plt.close()

def plot_sd():
    # plot sd matrix at every max ent iteration
    dataset = 'dataset_09_30_22'
    dir = f'/home/erschultz/{dataset}/samples'
    sample = 10
    sample_dir = osp.join(dir, f'sample{sample}')
    y = np.load(osp.join(sample_dir, 'y.npy'))
    y /= np.max(y)
    s = np.load(osp.join(sample_dir, 's.npy'))
    s_sym = (s + s.T)/2
    diag_chis = np.load(osp.join(sample_dir, 'diag_chis.npy'))
    d = calculate_D(diag_chis)
    sd = s_sym + d

    rep_dir = osp.join(sample_dir, 'PCA-normalize/k4/replicate1')
    chis = np.loadtxt(osp.join(rep_dir, 'chis.txt'))
    chis_diag = np.loadtxt(osp.join(rep_dir, 'chis_diag.txt'))
    x = np.load(osp.join(rep_dir, 'resources/x.npy'))
    m, k = x.shape

    rmse_list = []
    rmse_diag_list = []
    rmse_s_list = []
    rmse_y_list = []
    for it in range(1, 22):
        print(it)
        it_dir = osp.join(rep_dir, f'iteration{it}')
        chis_it = triu_to_full(chis[it])
        chis_diag_it = chis_diag[it]

        data_file = osp.join(it_dir, 'production_out.tar.gz')
        if osp.exists(data_file):
            with tarfile.open(data_file) as f:
                f.extractall(it_dir)
        y_file = osp.join(it_dir, 'production_out/contacts.txt')
        yhat = np.loadtxt(y_file)
        yhat /= np.max(yhat)
        rmse = mean_squared_error(y, yhat, squared = False)
        rmse_y_list.append(rmse)

        s_it = calculate_S(x, chis_it)
        s_sym_it = (s_it + s_it.T)/2
        rmse_s_list.append(mean_squared_error(s_sym, s_sym_it, squared = False))

        with open(osp.join(it_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        diag_chi_continuous = calculate_diag_chi_step(config, chis_diag_it)
        rmse_diag_list.append(mean_squared_error(diag_chi_continuous, diag_chis,
                                                squared = False))

        d = calculate_D(diag_chi_continuous)
        sd_it = s_sym_it + d
        rmse_list.append(mean_squared_error(sd, sd_it, squared = False))

    plt.plot(rmse_list, label = 'combined')
    plt.plot(rmse_diag_list, label = 'diag')
    plt.plot(rmse_s_list, label = 's')
    plt.plot(rmse_y_list, label = 'y')
    plt.ylabel('RMSE')
    plt.xlabel('Max Ent Iteration')
    plt.yscale('log')
    plt.legend()
    plt.savefig(osp.join(rep_dir, 'convergence_mse.png'))
    plt.show()

def compare_y_diag():
    dir = '/home/erschultz/dataset_test/samples'

    #
    # for sample in [1, 2, 3, 4]:
    #     sample_dir = osp.join(dir, f'sample{sample}')
    #     y_ticg = np.load(osp.join(sample_dir, 'y.npy'))
    #     y_diag_ticg = np.load(osp.join(sample_dir, 'y_diag.npy'))
    #     for sweep in [100000, 250000, 500000, 750000, 1000000]:
    #         y = np.loadtxt(osp.join(sample_dir, 'data_out', f'contacts{sweep}.txt'))
    #         np.save(osp.join(sample_dir, f'y_sweep{sweep}.npy'), y)
    #         plot_matrix(y, osp.join(sample_dir, f'y_sweep{sweep}.png'), vmax = 'mean')
    #
    #         diff = y_ticg - y
            # plot_matrix(diff, osp.join(sample_dir, f'y_vs_y_sweep{sweep}.png'),
            #             title = f'y - y_sweep{sweep}', cmap='bluered')
    #
    #         meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    #         y_diag = DiagonalPreprocessing.process(y, meanDist)
            # plot_matrix(y_diag, osp.join(sample_dir, f'y_diag_sweep{sweep}.png'),
            #             vmax = 'max')
    #
    #         diff = y_diag_ticg - y_diag
            # plot_matrix(diff, osp.join(sample_dir, f'y_diag_vs_y_diag_sweep{sweep}.png'),
            #             title = f'y_diag - y_diag_sweep{sweep}', cmap='bluered')

    mode='grid'
    for sample in [1, 2, 3, 4]:
        sample_dir = osp.join(dir, f'sample{sample}')
        y_ticg = np.load(osp.join(sample_dir, 'y.npy'))
        y_diag_ticg = np.load(osp.join(sample_dir, 'y_diag.npy'))
        for size in [1000, 2500, 5000, 10000]:
            y = np.loadtxt(osp.join(sample_dir, 'data_out', f'contacts{size}.txt'))
            np.save(osp.join(sample_dir, f'y_{mode}{size}.npy'), y)
            plot_matrix(y, osp.join(sample_dir, f'y_{mode}{size}.png'), vmax = 'mean')

            diff = y_ticg - y
            plot_matrix(diff, osp.join(sample_dir, f'y_vs_y_{mode}{size}.png'),
                        title = f'y - y_{mode}{size}', cmap='bluered')

            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
            y_diag = DiagonalPreprocessing.process(y, meanDist)
            plot_matrix(y_diag, osp.join(sample_dir, f'y_diag_{mode}{size}.png'), vmax = 'max')

            diff = y_diag_ticg - y_diag
            plot_matrix(diff, osp.join(sample_dir, f'y_diag_vs_y_diag_{mode}{size}.png'),
                        title = f'y_diag - y_diag_{mode}{size}', cmap='bluered')


    # dir = osp.join(dir, 'sample1')
    # y_diag = np.load(osp.join(dir, 'y_diag.npy'))
    # y_diag_log = np.log(y_diag)
    #
    # dir = osp.join(dir, 'sample2')
    # y2_diag = np.load(osp.join(dir, 'y_diag.npy'))
    # y2_diag_log = np.log(y1000_diag)
    #
    # diff = y_diag - y1000_diag
    # plot_matrix(diff, osp.join(dir, 'diagvs1000diag.png'), 'diag - 1000diag',
    #             vmin = 'min', vmax = 'max', cmap = 'bluered')

    # diff = y_diag - y5000_diag
    # plot_matrix(diff, osp.join(dir, 'diagvs5000diag.png'), 'diag - 5000diag',
    #             vmin = 'min', vmax = 'max', cmap = 'bluered')
    #
    # diff = y5000_diag - y1000_diag
    # plot_matrix(diff, osp.join(dir, 'diag5000vs1000diag.png'),
    #             '5000diag - 1000diag', vmin = 'min',
    #             vmax = 'max', cmap = 'bluered')
    #
    # diff = y_diag_log - y5000_diag_log
    # plot_matrix(diff, osp.join(dir, 'diaglogvs5000diaglog.png'),
    #             'diaglog - 5000diaglog', vmin = 'min',
    #             vmax = 'max', cmap = 'bluered')
    #
    # diff = y5000_diag_log - y1000_diag_log
    # plot_matrix(diff, osp.join(dir, 'diag5000logvs1000diaglog.png'),
    #             '5000diaglog - 1000diaglog', vmin = 'min',
    #             vmax = 'max', cmap = 'bluered')

def check_if_same():
    # check if contact maps are identical
    dir = '/home/erschultz/dataset_test/samples'

    y_file = 'y.npy'
    y_ref = np.load(osp.join(dir, f'sample301/{y_file}'))

    for s in [304, 305, 306, 307]:
        print('s', s)
        sample_dir = osp.join(dir, f'sample{s}')
        y = np.load(osp.join(sample_dir, f'{y_file}'))
        diff = y_ref - y
        print(np.mean(diff))
        plot_matrix(diff, osp.join(sample_dir, 'diff.png'), cmap='bluered')

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

def molar_contact_ratio(dataset, plot=True):
    dir = '/project2/depablo/erschultz/'
    if not osp.exists(dir):
        dir = '/home/erschultz'
    data_dir = osp.join(dir, dataset)
    odir = osp.join(data_dir, 'molar_contact_ratio')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    def c(y, a, b):
        return a@y@b

    def r(y, a, b):
        denom = c(y,a,b)**2
        num = c(y,a,a) * c(y,b,b)
        return num/denom

    def get_seq_kmeans():
        kmeans = KMeans(n_clusters = 2)
        kmeans.fit(y_diag)
        seq = np.zeros((m, 2))
        seq[np.arange(m), kmeans.labels_] = 1
        return seq

    def get_seq_pca(binarize=False):
        pca = PCA()
        pca.fit(y_diag)
        pc = pca.components_[0]
        # normalize


        pca_var[i] = pca.explained_variance_ratio_[0]
        pcpos = pc.copy()
        pcpos[pc < 0] = 0 # set negative part to zero
        pcneg = pc.copy()
        pcneg[pc > 0] = 0 # set positive part to zero
        pcneg *= -1 # make positive

        if binarize:
            val = np.mean(pcpos)
            pcpos[pcpos <= val] = 0
            pcpos[pcpos > val] = 1
            val = np.mean(pcneg)
            pcneg[pcneg <= val] = 0
            pcneg[pcneg > val] = 1

        seq[:,0] = pcpos
        seq[:,1] = pcneg

        return seq

    def get_gnn_mse():
        model_ID=341
        model_path = f'/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/{model_ID}'
        argparse_path = osp.join(model_path, 'argparse.txt')


        # set up argparse options
        parser = get_base_parser()
        sys.argv = [sys.argv[0]] # delete args from get_params, otherwise gnn opt will try and use them
        opt = parser.parse_args(['@{}'.format(argparse_path)])
        opt.id = int(model_ID)
        print(opt)
        opt = finalize_opt(opt, parser, local = True, debug = True)
        opt.data_folder = data_dir
        opt.root_name = f'GNN{opt.id}-test' # need this to be unique
        opt.log_file = sys.stdout # change
        opt.cuda = False # force to use cpu
        opt.device = torch.device('cpu')
        opt.verbose = False
        opt.scratch = '/home/erschultz/scratch'
        if opt.y_preprocessing.startswith('sweep'):
            _, *opt.y_preprocessing = opt.y_preprocessing.split('_')
            if isinstance(opt.y_preprocessing, list):
                opt.y_preprocessing = '_'.join(opt.y_preprocessing)
        print(opt.output_mode)

        # get model
        model, _, _ = load_saved_model(opt, False)

        # get dataset
        dataset = get_dataset(opt, verbose = False, samples = samples)
        print('Dataset: ', dataset, len(dataset))
        dataloader = torch_geometric.loader.DataLoader(dataset, batch_size = 1,
                            shuffle = False, num_workers = 1)

        # get prediction
        mse_dict = {}
        for i, data in enumerate(dataloader):
            data = data.to(opt.device)
            yhat = model(data)
            loss = opt.criterion(yhat, data.energy)
            mse_dict[int(osp.split(data.path[0])[1][6:])] = np.round(loss.item(), 3)
            yhat = yhat.cpu().detach().numpy().reshape((opt.m,opt.m))

            if opt.output_preprocesing == 'log':
                yhat = np.multiply(np.sign(yhat), np.exp(np.abs(yhat)) - 1)

        # cleanup
        # opt.root is set in get_dataset
        clean_directories(GNN_path = opt.root)

        return mse_dict

    experimental = False
    if dataset == 'dataset_12_20_22':
        samples = [324, 981, 1936, 2834, 3464]
    elif dataset == 'dataset_11_21_22':
        samples = [1, 2, 3, 410, 653, 1462, 1801, 2290]
        # samples = list(range(1,1000))
        # samples.extend(range(1021, 2401))
    elif dataset == 'dataset_11_14_22':
        samples = range(2201, 2222)
        experimental = True
    elif dataset == 'dataset_01_26_23':
        samples = range(201, 283)
        experimental = True
    elif dataset.startswith('dataset_01_27_23'):
        samples = range(1, 15)
        experimental = True
    else:
        samples = range(1, 11)
    samples = np.array(samples)

    N = len(samples)
    k_means_rab = np.zeros(N)
    pca_rab = np.zeros(N)
    pca_b_rab = np.zeros(N)
    pca_var = np.zeros(N)
    y_list = []
    L_list, _ = plaid_dist(dataset, 4, False)
    L_list_exp, _ = plaid_dist('dataset_01_26_23', 4, False)
    meanDist_list = []
    for i, sample in enumerate(samples):
        sample_dir = osp.join(data_dir, f'samples/sample{sample}')

        y, y_diag = load_Y(sample_dir)
        y /= np.mean(np.diagonal(y))
        y_list.append(y)

        meanDist_list.append(DiagonalPreprocessing.genomic_distance_statistics(y))

        m = len(y)

        # plot_seq_binary(seq, save = False, show = True)

        # kmeans
        seq = get_seq_kmeans()
        rab = r(y, seq[:, 1], seq[:, 0])
        k_means_rab[i] = rab

        # pca
        seq = get_seq_pca()
        rab = r(y, seq[:, 1], seq[:, 0])
        pca_rab[i] = rab

        # pca
        seq = get_seq_pca(True)
        rab = r(y, seq[:, 1], seq[:, 0])
        pca_b_rab[i] = rab

    # GNN MSE
    if not experimental:
        mse_dict = get_gnn_mse()
        mse_list = []
        for i, s in enumerate(samples):
            mse = mse_dict[s]
            mse_list.append(mse)
        np.save(osp.join(odir, 'mse.npy'), mse_list)

        # make table
        # with open(osp.join(odir, 'plaid_score_table.txt'), 'w') as o:
        #     o.write("\\begin{center}\n")
        #     o.write("\\begin{tabular}{|" + "c|"*5 + "}\n")
        #     o.write("\\hline\n")
        #     o.write("\\multicolumn{5}{|c|}{" + dataset.replace('_', "\_") + "} \\\ \n")
        #     o.write("\\hline\n")
        #     o.write('Sample & K\_means R(a,b) & PCA R(a,b) & PCA \% Var & MSE \\\ \n')
        #     o.write("\\hline\\hline\n")
        #     for i, mse in enumerate(mse_list):
        #         vals = [k_means_rab[i], pca_rab[i], pca_var[i]]
        #         vals = np.round(vals, 4)
        #         o.write(f'{s} & {vals[0]} & {vals[1]} & {vals[2]} & {mse}\\\ \n')
        #     o.write("\\hline\n")
        #     o.write("\\end{tabular}\n")
        #     o.write("\\end{center}\n\n")

        # make new dataset with samples < cutoff
        new_data_dir = osp.join(dir, f'{dataset}_plaid_cutoff')
        if osp.exists(new_data_dir):
            shutil.rmtree(new_data_dir)
        os.mkdir(new_data_dir, mode=0o755)
        os.mkdir(osp.join(new_data_dir, 'samples'), mode=0o755)
        cutoff = 50
        for i, s in enumerate(samples):
            if k_means_rab[i] < cutoff:
                shutil.copytree(osp.join(data_dir, f'samples/sample{s}'),
                            osp.join(new_data_dir, f'samples/sample{s}'))




    # plot distributions
    if plot:
        # plot histograms
        for arr, label in zip([k_means_rab, pca_rab, pca_b_rab, pca_var],
                                ['kmeans_Rab', 'PCA_Rab', 'PCA_binary_Rab','PCA_var']):
            np.save(osp.join(odir, label + '.npy'), arr)
            print(label)
            if not experimental:
                p = pearson_round(arr, mse_list)
                print(p)
            arr = np.array(arr).reshape(-1)
            print(np.min(arr), np.max(arr))
            n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = 50,
                                        alpha = 0.5, label = label)
            plt.legend()
            plt.ylabel('probability', fontsize=16)
            plt.xlabel(f'{label}', fontsize=16)
            plt.xscale('log')
            plt.savefig(osp.join(odir, f'{label}_distribution.png'))
            plt.close()


        # crop samples for plotting
        plot_n = 8
        y_arr = np.array(y_list[:8])
        L_arr = np.array(L_list[:8])
        k_means_rab = k_means_rab[:8]
        meanDist_arr = np.array(meanDist_list[:8])
        samples = samples[:8]

        # plot contact maps orderd by rab
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0,    'white'),
                                                  (1,    'red')], N=126)
        ind = np.argsort(k_means_rab)
        y_arr = np.array(y_list)
        fig, ax = plt.subplots(2, 5,
                                gridspec_kw={'width_ratios':[1,1,1,1,0.08]})
        fig.set_figheight(6*2)
        fig.set_figwidth(6*3)
        vmin = 0; vmax = np.mean(y_arr)
        row = 0; col=0
        for y, val, sample in zip(y_arr[ind], k_means_rab[ind], samples[ind]):
            if col == 0:
                s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap,
                    ax = ax[row][col], cbar_ax = ax[row][4])
            else:
                s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap,
                    ax = ax[row][col], cbar = False)
            s.set_title(f'Sample {sample}\nPlaid Score = {np.round(val, 1)}', fontsize = 16)
            s.set_xticks([])
            s.set_yticks([])

            col += 1
            if col == 4:
                col = 0
                row += 1

        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'y_ordered.png'))
        plt.close()


        # plot L_ij ordered by rab
        fig, ax = plt.subplots(2, 4)
        fig.set_figheight(6*2)
        fig.set_figwidth(6*3)
        row = 0; col=0
        bin_width = 1
        arr_exp = np.array(L_list_exp).reshape(-1)
        for L, val, sample in zip(L_arr[ind], k_means_rab[ind], samples[ind]):
            arr = L.reshape(-1)
            bins = range(math.floor(min(arr_exp)), math.ceil(max(arr_exp)) + bin_width, bin_width)
            ax[row][col].hist(arr_exp, weights = np.ones_like(arr_exp) / len(arr_exp),
                                        bins = bins,
                                        alpha = 0.5, label = 'Experiment')
            bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
            ax[row][col].hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = bins,
                                        alpha = 0.5, label = 'Simulation')
            ax[row][col].set_title(f'Sample {sample}\nPlaid Score = {np.round(val, 1)}', fontsize = 16)

            col += 1
            if col == 4:
                col = 0
                row += 1

        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'L_dist_ordered.png'))
        plt.close()

        # plot meanDist colored by plaid score
        for meanDist, val, sample in zip(meanDist_arr[ind], k_means_rab[ind], samples[ind]):
            plt.plot(meanDist, label = np.round(val, 1))
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'meanDist_plaid_score.png'))
        plt.close()

    return meanDist_list

def meanDist_comparison():
    data_dir = '/home/erschultz/dataset_01_26_23'
    meanDist_exp_list = molar_contact_ratio('dataset_01_26_23', False)
    meanDist_max_ent_list = molar_contact_ratio('dataset_01_27_23_v9', False)

    fig, ax = plt.subplots()
    for meanDist in meanDist_exp_list:
        ax.plot(meanDist, c = 'k')
    for meanDist in meanDist_max_ent_list:
        ax.plot(meanDist,c = 'b')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax2 = ax.twinx()
    ax2.plot(np.NaN, np.NaN, label = 'Experiment', c = 'k')
    ax2.plot(np.NaN, np.NaN, label = 'Max Ent', c = 'b')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(osp.join(data_dir, 'meanDist_comparison.png'))
    plt.close()

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

def main():
    dir = '/home/erschultz/dataset_11_21_22/molar_contact_ratio_cluster'
    def simple_plot(arr, fname):
        print(np.min(arr), np.max(arr))
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = np.logspace(np.log10(1),np.log10(1000), 50),
                                    alpha = 0.5)
        plt.ylabel('probability', fontsize=16)
        plt.xscale('log')
        plt.savefig(osp.join(dir, fname))
        plt.close()


    mse_arr = np.load(osp.join(dir, 'mse.npy'))
    simple_plot(mse_arr, 'mse_distribution.png')

    kmeans_arr = np.load(osp.join(dir, 'kmeans_Rab.npy'))
    simple_plot(kmeans_arr, 'kmeans_distribution.png')
    result = pearson_round(kmeans_arr, mse_arr, stat = 'spearman')
    print(result)

    # plt.scatter(kmeans_arr, mse_arr)
    # plt.xlim(0, 100)
    # plt.show()

    kmeans_exp = np.load('/home/erschultz/dataset_01_26_23/molar_contact_ratio/kmeans_Rab.npy')
    for arr, label in zip([kmeans_arr, kmeans_exp], ['Synthetic', 'Experiment']):
        print(label)
        print(np.min(arr), np.max(arr))
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = np.logspace(np.log10(1),np.log10(1000), 50),
                                    alpha = 0.5, label = label)
    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel('KMeans Plaid Score', fontsize=16)
    plt.xscale('log')
    plt.savefig(osp.join(dir, f'kmeans_vs_exp_distribution.png'))
    plt.close()

def temp_p_s():
    dir = '/home/erschultz/dataset_01_26_23/samples/sample265'
    y_exp = np.load(osp.join(dir, 'y.npy'))
    y_g = np.load(osp.join(dir, 'none/k1/replicate1/y.npy'))
    y_me = np.load(osp.join(dir, 'PCA-normalize-E/k4/replicate1/y.npy'))
    max_it = get_final_max_ent_folder(osp.join(dir, 'PCA-normalize-E/k4/replicate1'))
    with open(osp.join(max_it, 'config.json')) as f:
        config = json.load(f)
        diag_chis_step = calculate_diag_chi_step(config)

    fig, ax = plt.subplots()
    for y, label, c in zip([y_exp, y_me, y_g], ['Experiment', 'Max Ent', 'Gaussian Chain'], ['black', 'blue', 'grey']):
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        ax.plot(meanDist, label = label, color = c)

    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.set_xscale('log')

    if diag_chis_step is not None:
        ax2 = ax.twinx()
        ax2.plot(diag_chis_step, ls = '--', label = 'Parameters', color = 'blue')
        ax2.set_ylabel('Diagonal Parameter', fontsize = 16)
        ax2.set_xscale('log')
        ax2.legend(loc='upper right')

    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'meanDist_comparison.png'))
    plt.close()


if __name__ == '__main__':
    # compare_y_diag()
    # check_if_same()
    # test_robust_PCA()
    # check_dataset('dataset_11_18_22')
    # time_comparison()
    # time_comparison_dmatrix()
    # construct_sc_xyz()
    # convergence_check()
    # main()
    # plot_p_s()
    # molar_contact_ratio('dataset_01_27_23_v5', True)
    # molar_contact_ratio('dataset_01_26_23', True)
    # molar_contact_ratio('dataset_01_27_23_v9', True)
    # compare_scc_bio_replicates()
    # plot_sd()
    # max_ent_loss_for_gnn('dataset_11_14_22', 2201)
    # temp_p_s()
    # meanDist_comparison()
    plot_p_s('dataset_test', params=True)
