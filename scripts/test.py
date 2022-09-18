import json
import math
import os
import os.path as osp
from collections import defaultdict

import hicrep
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from seq2contact import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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
                    seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
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

def makeDirsForMaxEnt(dataset):
    '''
    Create empty directories with appropriate max ent file stucture.
    Helpful for before downloading data from cluster.
    '''
    for sample in [1]:
        dir = '/home/erschultz/sequences_to_contact_maps'
        sample_folder = osp.join(dir, dataset, 'samples', f'sample{sample}')
        assert osp.exists(sample_folder)

        for method in ['PCA-normalize']:
            os.mkdir(osp.join(sample_folder, method), mode = 0o755)
            for k in [2, 4, 6]:
                os.mkdir(osp.join(sample_folder, method, f'k{k}'), mode = 0o755)
                for replicate in [1]:
                    os.mkdir(osp.join(sample_folder, method, f'k{k}', f'replicate{replicate}'), mode = 0o755)

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
        # plotContactMap(m_log, ofile = osp.join(dir, 'M_log.png'), vmin = 'min', vmax = 'max')
        # plotContactMap(s0, ofile = osp.join(dir, 'S0.png'), vmin = 'min', vmax = 'mean', cmap='blue-red')

        # plotContactMap(p, ofile = osp.join(dir, 'P.png'), vmax = 'mean')
        # plotContactMap(p_log, ofile = osp.join(dir, 'P_log.png'), vmin = 'min', vmax = 'max')

        # L, S = R_pca(m).fit(max_iter=200)
        # plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = 'mean')
        # plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmin = 'min', vmax = 'max', cmap='blue-red')
        L_log, S_log = R_pca(m_log).fit(max_iter=2000)
        plotContactMap(L_log, ofile = osp.join(dir, 'RPCA_L_log.png'), vmin = 'min', vmax = 'max')
        plotContactMap(S_log, ofile = osp.join(dir, 'RPCA_S_log.png'), vmin = 'min', vmax = 'max', cmap='blue-red')
        L_log_exp = np.exp(L_log)
        S_log_exp = np.exp(S_log)
        plotContactMap(L_log_exp, ofile = osp.join(dir, 'RPCA_L_log_exp.png'), vmax = 'max')
        plotContactMap(S_log_exp, ofile = osp.join(dir, 'RPCA_S_log_exp.png'), vmin = 'min', vmax = 'max', cmap='blue-red')

        PC_m = plot_top_PCs(m, inp_type='m', verbose = True, odir = dir, plot = True)
        meanDist = genomic_distance_statistics(m)
        m_diag = diagonal_preprocessing(m, meanDist)
        PC_m_diag = plot_top_PCs(m_diag, inp_type='m_diag', verbose = True, odir = dir, plot = True)

        # plot_top_PCs(L_log_exp, inp_type='L_log_exp', verbose = True, odir = dir, plot = True)
        PC_L_log = plot_top_PCs(L_log, inp_type='L_log', verbose = True, odir = dir, plot = True)
        stat = pearsonround(PC_L_log[0], PC_m[0])
        print("Correlation between PC 1 of L_log and M: ", stat)
        stat = pearsonround(PC_L_log[1], PC_m[1])
        print("Correlation between PC 2 of L_log and M: ", stat)
        meanDist = genomic_distance_statistics(L_log)
        L_log_diag = diagonal_preprocessing(L_log, meanDist)
        PC_L_log_diag = plot_top_PCs(L_log_diag, inp_type='L_log_diag', verbose = True, odir = dir, plot = True)
        stat = pearsonround(PC_L_log_diag[0], PC_m_diag[0])
        print("Correlation between PC 1 of L_log_diag and M_diag: ", stat)
        stat = pearsonround(PC_L_log_diag[1], PC_m_diag[1])
        print("Correlation between PC 2 of L_log_diag and M_diag: ", stat)
        # plot_top_PCs(m_log, inp_type='m_log', verbose = True, odir = dir, plot = True)
        # meanDist = genomic_distance_statistics(m_log)
        # m_log_diag = diagonal_preprocessing(m_log, meanDist)
        # plot_top_PCs(m_log_diag, inp_type='m_log_diag', verbose = True, odir = dir, plot = True)


        # ydiag = np.load(osp.join(dataset_test, 'sample21/y_diag.npy'))
        # ydiag_rank_1 = np.load(osp.join(dataset_test, 'sample21/PCA_analysis/y_diag_rank_1.npy'))
        # dif = ydiag - ydiag_rank_1
        # plotContactMap(ydiag, ofile = osp.join(dir, 'ydiag.png'), vmax = 'max')
        # plotContactMap(ydiag_rank_1, ofile = osp.join(dir, 'ydiag_rank_1.png'), vmax = 'max')
        # plotContactMap(dif, ofile = osp.join(dir, 'dif.png'), vmin = 'min', vmax = 'max', cmap='blue-red')


    if False:
        # dir = '/home/eric/dataset_test/samples/sample104'
        dir = '/home/eric/sequences_to_contact_maps/dataset_09_21_21/samples/sample1'
        y = np.load(osp.join(dir, 'y.npy'))
        # L, S = R_pca(y).fit(max_iter=200)
        # plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = np.mean(y))
        # plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmax = np.mean(y))

        # l = 1/np.sqrt(1024) * 1/100
        y_diag = np.load(osp.join(dir, 'y_diag.npy'))
        L, S = R_pca(y_diag).fit(max_iter=200)
        plotContactMap(L, ofile = osp.join(dir, 'RPCA_L_diag.png'), vmin='min', vmax = 'max')
        plotContactMap(S, ofile = osp.join(dir, 'RPCA_S_diag.png'), vmin='min', vmax = np.max(S), cmap='blue-red')
        plotContactMap(y_diag, ofile = osp.join(dir, 'y_diag.png'), vmax = 'max')
        plot_top_PCs(L, verbose = True)

def time_comparison():
    # dir = '/project2/depablo/erschultz/dataset_05_18_22/samples'
    dir = '/home/erschultz/sequences_to_contact_maps/dataset_05_18_22/samples'
    samples_per_size = 3
    num_sizes = 3

    times_dict = defaultdict(lambda: np.full([num_sizes, samples_per_size], np.nan))
    # dictionary with keys = method : vals = array of times with rows = sizes, cols = replicate samples
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

def time_comparison_merge_PCA():
    # dir = '/project2/depablo/erschultz/dataset_05_18_22/samples'
    dir = '/home/erschultz/sequences_to_contact_maps/dataset_05_18_22/samples'
    samples_per_size = 3
    num_sizes = 3

    times_dict = defaultdict()
    # dictionary with keys = method : vals = array of times with rows = sizes, cols = replicate samples
    it = 1
    for row in range(num_sizes):
        PCA_col = 0
        PCA_diag_col = 0
        for col in range(samples_per_size):
            print(it)
            sample_dir = osp.join(dir, f'sample{it}')
            max_ent_file = osp.join(sample_dir, 'max_ent_table_2.txt')

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
                        label = f'{method}'
                        print(label)
                        time = float(line_list[-1].split(' ')[0])
                    except:
                        print(line)
                        raise
                    if label not in times_dict:
                        if 'PCA' in label:
                            times_dict[label] =  np.full([num_sizes, samples_per_size*3], np.nan)

                        else:
                            times_dict[label] =  np.full([num_sizes, samples_per_size], np.nan)

                    if 'PCA-diagMLP' in label:
                        print(it, k, PCA_diag_col)
                        times_dict[label][row, PCA_diag_col] = time
                        PCA_diag_col += 1
                    elif 'PCA' in label:
                        print(it, k, PCA_col)
                        times_dict[label][row, PCA_col] = time
                        PCA_col += 1
                    else:
                        times_dict[label][row,col] = time
                    line = f.readline()

            it += 1


    cmap = matplotlib.cm.get_cmap('tab20')
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    sizes = np.array([512., 1024., 2048., 4096.])[:num_sizes]
    sizes_shift = np.copy(sizes)
    ind = 0
    for method in sorted(times_dict.keys()):
        if method in {'GNN-'}:
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

        ax2.plot(sizes_shift, prcnt_converged, ls = '--', color = cmap(ind % cmap.N))
        ax.errorbar(sizes_shift, times, yerr = times_std, label = method,
                    color = cmap(ind % cmap.N), fmt = "o")
        sizes_shift += 20
        ind += 1


    ax.set_ylabel('Total Time (mins)', fontsize=16)
    ax2.set_ylabel(f'Percent Converged (of {arr.shape[1]})', fontsize=16)
    ax.set_xlabel('Simulation size', fontsize=16)
    ax.set_ylim((0, None))
    ax.set_xticks(sizes) # , 2048, 4096
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'time_merge.png'))
    plt.close()

def convergence_check():
    dir = '/home/erschultz/sequences_to_contact_maps/dataset_05_18_22/samples'
    results_1 = {}
    results_2 = {}
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


                                conv_file = osp.join(replicate, 'convergence_diag.txt')
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
        if id < 7:
            continue
        k = int(label[-4])

        plt.plot(vals, label = label, ls = ls_arr[k // 2], color = cmap(id % cmap.N))

    plt.axhline(5, c = 'k')
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel(r'$||x_t - x_{t-1}||_2$', fontsize=16)
    plt.legend()
    plt.title(r'$||x_t - x_{t-1}||_2 < 5$')
    plt.savefig(osp.join(dir, 'param_convergence.png'))
    plt.close()


    cmap = matplotlib.cm.get_cmap('tab10')
    ls_arr = ['solid', 'dotted', 'dashed', 'dashdot']
    for label, vals in results_2.items():
        print(label)
        id = int(label.split('_')[0])
        if id < 7:
            continue
        k = int(label[-4])

        plt.plot(vals, label = label, ls = ls_arr[k // 2], color = cmap(id % cmap.N))

    plt.axhline(1e-2, c = 'k')
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel(r'$|f(x_t) - f(x_{t-1})|$', fontsize=16)
    plt.legend(loc = 'upper right')
    plt.title(r'$|f(x_t) - f(x_{t-1})| < 10^{-2}$')
    plt.savefig(osp.join(dir, 'loss_convergence.png'))
    plt.close()

def bin_zhang_contact_function_comparison():
    x = np.linspace(0, 5, 1000)
    y = 0.5 * (1 + np.tanh(3.22*(1.78 - x)))
    plt.plot(x, y, label=2016)
    x = np.linspace(0, 1.76, 1000)
    y = 0.5 * (1 + np.tanh(3.72*(1.76 - x)))
    plt.plot(x, y, label = 2019, color = 'k')
    x = np.linspace(1.76, 5, 1000)
    y = 0.5 * (1.76 / x)**4
    plt.plot(x, y, color = 'k')
    plt.legend()
    plt.show()

def main():
    # quick function to plot sc p(s) curves
    dir = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017'
    data_dir = osp.join(dir, 'contact_diffusion_kNN8scc/iteration_1/sc_contacts')
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    with open(osp.join(dir, 'contact_diffusion_kNN8scc/iteration_0/sc_contacts/ifile_dict.json')) as f:
        ifile_dict = json.load(f)

    with open(osp.join(dir, 'samples/phase_dict.json'), 'r') as f:
        phase_dict = json.load(f)

    phase_count_dict = defaultdict(int) # how many times have we seen each phase
    for file in os.listdir(data_dir):
        if file.endswith('cool'):
            ifile = osp.join(data_dir, file)
            i = file.split('.')[0].split('_')[-1]

            dir = ifile_dict[osp.split(file)[1].replace('.cool', '.mcool')]
            phase = phase_dict[dir]
            phase_count_dict[phase] += 1

            if phase_count_dict[phase] < 2:
                print(file)
                y = load_contact_map(ifile, chrom=15, resolution=50000)

                ofile = osp.join(data_dir, 'sc_contacts_time', f'y_sc_{i}_chrom15.png')
                contacts = int(np.sum(y) / 2)
                sparsity = np.round(np.count_nonzero(y) / len(y)**2 * 100, 2)
                title = f'Sample {i}:\n# contacts: {contacts}, sparsity: {sparsity}%'
                plot_matrix(y, ofile, title, vmax = 'mean')

                meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
                print(meanDist)
                ax.plot(meanDist, label = phase)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    ax.legend()
    plt.tight_layout()
    plt.savefig(osp.join(data_dir, 'sc_contacts_time', 'meanDist_log2.png'))
    plt.close()

def main2(sample=244):
    # plot different p(s) curves
    dir = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_imputed/samples'
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ifile = osp.join(dir, f'sample{sample}/y.npy')
    y_gt = np.load(ifile)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob')
    ax.plot(meanDist, label = 'Experiment' , color = 'k')

    # ifile = osp.join(dir, f'sample{sample}/none-diagMLP-41/k0/replicate1/y.npy')
    # y = np.load(ifile)
    # meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    # ax.plot(meanDist, label='MLP', color = 'green')

    ifile = osp.join(dir, f'sample{sample}/none/k0/replicate1/y.npy')
    y_max_ent = np.load(ifile)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_max_ent, 'prob')
    ax.plot(meanDist, label = 'max_ent', color = 'blue')

    ifile = osp.join(dir, f'sample{sample}_edit/y.npy')
    y = np.load(ifile)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    ax.plot(meanDist, label = 'max_ent_adjusted', color = 'lightblue')


    # ifile = osp.join(dir, f'sample443_log/y.npy')
    # y = np.load(ifile)
    # meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    # ax.plot(meanDist, label = 'log_fit')
    #
    # ifile = osp.join(dir, f'sample443_logistic/y.npy')
    # y = np.load(ifile)
    # meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    # ax.plot(meanDist, label = 'logistic_fit')


    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    ax.legend()
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'sample{sample}_edit', 'meanDist_log.png'))
    plt.close()

def modify_maxent_diag_chi(sample = 443):
    # try different modifications to diag chis learned by max ent
    # dir = f'/home/erschultz/sequences_to_contact_maps/single_cell_nagano_imputed/samples/sample{sample}/none/k0/replicate1'
    # ifile = osp.join(dir, 'chis_diag.txt')
    # diag_chis = np.loadtxt(ifile)[-1]
    dir = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_imputed/samples/sample443_edit'
    diag_chis = np.load(osp.join(dir, 'diag_chis.npy'))

    ifile = osp.join(dir, 'config.json')
    with open(ifile, 'r') as f:
        config = json.load(f)

    diag_chi_step = get_diag_chi_step(config, diag_chis)
    print(diag_chi_step[:12])

    x = np.arange(0, len(diag_chi_step))

    init = [1, 1, 0]
    popt, pcov = curve_fit(log_curve, x[8:60], diag_chi_step[8:60], p0 = init, maxfev = 2000)
    print('popt', popt)
    log_fit = log_curve(x, *popt)
    np.savetxt(osp.join(dir, 'log_fit.txt'), log_fit)



    init = [1, 0, 1, 0]
    popt, pcov = curve_fit(logistic_curve, x, diag_chi_step, p0 = init, maxfev = 2000)
    print('popt', popt)
    logistic_fit = logistic_curve(x, *popt)
    np.savetxt(osp.join(dir, 'logistic_fit.txt'), logistic_fit)


    plt.plot(diag_chi_step, label = 'diag_chi')
    plt.plot(log_fit, label = 'log')
    plt.plot(logistic_fit, label = 'logistic')
    plt.legend()
    plt.savefig(osp.join(dir, 'meanDist_fit.png'))
    plt.close()

    plt.plot(diag_chi_step, label = 'diag_chi')
    plt.plot(log_fit, label = 'log')
    plt.plot(logistic_fit, label = 'logistic')
    plt.xscale('log')
    plt.legend()
    plt.savefig(osp.join(dir, 'meanDist_fit_log.png'))
    plt.close()





def log_curve(x, A, B, C):
    result = A * np.log(B * x + 1) + C
    return result

def logistic_curve(x, max, min, slope, midpoint):
    result = (max - min)/ (1 + np.exp(-1*slope * (x + midpoint))) + min
    return result

def test_log_curve():
    x = np.arange(1024)


    y = log_curve(x, 15, 10/1000, -15)

    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # test_robust_PCA()
    # check_dataset('dataset_05_12_22')    # ifile = osp.join(dir, 'sample443_log/y.npy')
    # y = np.load(ifile)
    # meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    # ax.plot(meanDist, label = 'log_fit')
    # ifile = osp.join(dir, 'sample443_logistic/y.npy')
    # y = np.load(ifile)
    # meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    # ax.plot(meanDist, label = 'logistic_fit')
    # time_comparison()
    # time_comparison_merge_PCA()
    # construct_sc_xyz()
    # convergence_check()
    main2()
    # modify_maxent_diag_chi()
    # test_log_curve()
    # makeDirsForMaxEnt("dataset_09_21_21")
