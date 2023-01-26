import json
import math
import os
import os.path as osp
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ECDF import Ecdf
from get_params import Tester
from MultivariateSkewNormal import multivariate_skewnorm
from scipy.optimize import curve_fit
from scipy.stats import (beta, gamma, multivariate_normal, norm, skewnorm,
                         weibull_max, weibull_min)
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.energy_utils import (
    calculate_D, calculate_diag_chi_step, calculate_SD)
from sequences_to_contact_maps.scripts.plotting_utils import plot_matrix
from sequences_to_contact_maps.scripts.utils import (DiagonalPreprocessing,
                                                     triu_to_full)


def modify_maxent_diag_chi(sample, k = 8, edit = True):
    '''
    Inputs:
        sample: sample id
        k: number of marks
        edit: True to modify maxent result so that is is flat at start
    '''
    print(f'sample{sample}, k{k}')
    # try different modifications to diag chis learned by max ent
    dataset = 'dataset_11_14_22'
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    max_ent_dir = osp.join(dir, f'PCA-normalize-E/k{k}/replicate1')
    if not osp.exists(max_ent_dir):
        return
    if edit:
        odir = osp.join(max_ent_dir, 'fitting')
    else:
        odir = osp.join(max_ent_dir, 'fitting2')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    diag_chis = np.loadtxt(osp.join(max_ent_dir, 'chis_diag.txt'))[-1]

    if edit:
        # make version that is flat at start
        min_val = np.min(diag_chis)
        min_ind = np.argmin(diag_chis)
        diag_chis[0:min_ind] = min_val
        np.savetxt(osp.join(odir, 'chis_diag_edit.txt'), diag_chis)

        # make version with intercept 0
        diag_chis_zero = np.copy(diag_chis)
        diag_chis_zero -= np.min(diag_chis_zero)
        np.savetxt(osp.join(odir, 'chis_diag_edit_zero.txt'), diag_chis_zero)


    ifile = osp.join(max_ent_dir, 'resources/config.json')
    with open(ifile, 'r') as f:
        config = json.load(f)

    diag_chi_step = calculate_diag_chi_step(config, diag_chis)

    m = len(diag_chi_step)
    x = np.arange(0, 2*m)

    if edit:
        log_fit = curve_fit_helper(log_curve, x[:64], diag_chi_step[:64],
                                        'log', odir)
        log_max_fit = curve_fit_helper(log_max_curve, x[:64], diag_chi_step[:64],
                                        'log_max', odir)
        logistic_fit = curve_fit_helper(logistic_curve, x[:m], diag_chi_step,
                                        'logistic', odir, [10, 1, 100])
        poly2_fit = curve_fit_helper(poly2_curve, x[:m], diag_chi_step,
                                        'poly2', odir, [1, 1, 1])
        poly3_fit = curve_fit_helper(poly3_curve, x[:m], diag_chi_step,
                                        'poly3', odir, [1, 1, 1, 1])
        linear_max_fit = curve_fit_helper(linear_max_curve, x[:m], diag_chi_step,
                                        'linear_max', odir)
        linear_fit = curve_fit_helper(linear_curve, x[:m], diag_chi_step,
                                        'linear', odir)
    else:
        log_fit = None
        log_max_fit = None
        logistic_fit = None
        poly2_fit = curve_fit_helper(poly2_curve, x[:m], diag_chi_step,
                                        'poly2', odir, [1, 1, 1],
                                        start = 2)
        poly3_fit = curve_fit_helper(poly3_curve, x[:m], diag_chi_step,
                                        'poly3', odir, [1, 1, 1, 1],
                                        start = 2)
        piecewise_linear_fit = curve_fit_helper(piecewise_linear_curve, x[:m],
                                diag_chi_step, 'piecewise_linear', odir,
                                [1, 1, 1, 1, 10], start = 2)
        piecewise_poly2_fit = curve_fit_helper(piecewise_poly2_curve, x[:m],
                                diag_chi_step, 'piecewise_poly2', odir,
                                [1, 1, 1, 1, 1, 1, 10], start = 2)
        piecewise_poly3_fit = curve_fit_helper(piecewise_poly3_curve, x[:m],
                                diag_chi_step, 'piecewise_poly3', odir,
                                [1, 1, 1, 1, 1, 1, 1, 1, 20], start = 2)
        linear_max_fit = None
        linear_fit = None


    for log in [True, False]:
        plt.plot(diag_chi_step, label = 'Max Ent Edit', color = 'lightblue')
        if log_fit is not None:
            plt.plot(log_fit, label = 'log', color = 'orange')
        # if log_max_fit is not None:
        #     plt.plot(log_max_fit, label = 'log_max', color = 'yellow')
        if logistic_fit is not None:
            plt.plot(logistic_fit, label = 'logistic', color = 'purple')
        # if linear_max_fit is not None:
        #     plt.plot(linear_max_fit, label = 'linear_max', color = 'brown')
        if linear_fit is not None:
            plt.plot(linear_fit, label = 'linear', color = 'teal')
        if piecewise_linear_fit is not None:
            plt.plot(piecewise_linear_fit, label = 'piecewise_linear', color = 'teal', ls='--')
        if poly2_fit is not None:
            plt.plot(poly2_fit, label = 'poly2', color = 'pink')
        if piecewise_poly2_fit is not None:
            plt.plot(piecewise_poly2_fit, label = 'piecewise_poly2', color = 'pink', ls='--')
        if poly3_fit is not None:
            plt.plot(poly3_fit, label = 'poly3', color = 'red')
        if piecewise_poly3_fit is not None:
            plt.plot(piecewise_poly3_fit, label = 'piecewise_poly3', color = 'red', ls='--')
        plt.legend()
        plt.ylim(None, 2 * np.max(diag_chi_step))
        if log:
            plt.xscale('log')
            plt.savefig(osp.join(odir, 'meanDist_fit_log.png'))
        else:
            plt.xlim(0, 50)
            plt.savefig(osp.join(odir, 'meanDist_fit.png'))
        plt.close()

def curve_fit_helper(fn, x, y, label, odir, init = [1,1], start = 0):
    try:
        popt, pcov = curve_fit(fn, x[start:], y[start:], p0 = init, maxfev = 2000)
        print(f'\t{label} popt', popt)
        fit = fn(x, *popt)
        np.savetxt(osp.join(odir, f'{label}_fit.txt'), fit)
        np.savetxt(osp.join(odir, f'{label}_popt.txt'), popt)
    except RuntimeError as e:
        fit = None
        print(f'{label}:', e)

    return fit

def piecewise_linear_curve(x, m1, m2, b1, b2, midpoint):
    result = np.zeros_like(x).astype(np.float64)
    midpoint = int(midpoint)
    result[:midpoint] = x[:midpoint] * m1 + b1
    result[midpoint:] = x[midpoint:] * m2 + b2
    return result

def piecewise_poly2_curve(x, A1, B1, C1, A2, B2, C2, midpoint):
    result = np.zeros_like(x).astype(np.float64)
    midpoint = int(midpoint)
    result[:midpoint] = A1 + B1 * x[:midpoint] + C1 * x[:midpoint]**2
    result[midpoint:] = A2 + B2 * x[midpoint:] + C2 * x[midpoint:]**2
    return result

def piecewise_poly3_curve(x, A1, B1, C1, D1, A2, B2, C2, D2, midpoint):
    result = np.zeros_like(x).astype(np.float64)
    midpoint = int(midpoint)
    result[:midpoint] = A1 + B1 * x[:midpoint] + C1 * x[:midpoint]**2 + D1 * x[:midpoint]**3
    result[midpoint:] = A2 + B2 * x[midpoint:] + C2 * x[midpoint:]**2 + D2 * x[midpoint:]**3
    return result

def linear_curve(x, A, B):
    result = A*x + B
    return result

def linear_max_curve(x, A, B):
    result = A*x + B
    result[result < 0] = 0
    return result

def log_curve(x, A, B):
    result = A * np.log(B * x + 1)
    return result

def log_max_curve(x, A, B):
    result = A * np.log(B * x)
    result[result < 0] = 0
    return result

def logistic_curve(x, max, slope, midpoint):
    result = (max) / (1 + np.exp(-1*slope * (x - midpoint)))
    return result

def logistic_manual_curve(xmax, slope, midpoint):
    x, max = xmax
    return logistic_curve(x, max, slope, midpoint)

def poly3_curve(x, A, B, C, D):
    result = A + B*x + C*x**2 + D*x**3
    return result

def poly2_curve(x, A, B, C):
    result = A + B*x + C*x**2
    return result

def plot_modified_max_ent(sample, params = True, k = 8):
    # plot different p(s) curves
    print(f'sample{sample}, k{k}')
    dataset = 'dataset_11_14_22'
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    max_ent_dir = osp.join(dir, f'PCA-normalize-E/k{k}/replicate1')
    fig, ax = plt.subplots()
    if params:
        ax2 = ax.twinx()

    ifile = osp.join(dir, 'y.npy')
    y_gt = np.load(ifile)
    meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob', smoothen = False)
    ax.plot(meanDist_gt, label = 'Experiment' , color = 'k')

    ifile = osp.join(max_ent_dir, 'y.npy')
    y_maxent = np.load(ifile)
    meanDist_maxent = DiagonalPreprocessing.genomic_distance_statistics(y_maxent, 'prob', smoothen = False)
    ax.plot(meanDist_maxent, label = 'Max Ent' , color = 'blue')
    if params:
        # find config
        config = None
        config_file = osp.join(max_ent_dir, 'resources/config.json')
        if osp.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)

        # find diag_chis
        diag_chis = np.loadtxt(osp.join(max_ent_dir, 'chis_diag.txt'))[-1]

        diag_chis_step = calculate_diag_chi_step(config, diag_chis)
        ax2.plot(diag_chis_step, ls = '--', label = 'Parameters', color = 'blue')

    methods = [
             # f'sample{sample}/none-diagMLP-79/k0/replicate1',
            # f'sample{sample}/GNN-177-S-diagMLP-79/k0/replicate1',
            'edit', 'edit_zero',
            'zero', f'log', f'logistic',
            'log_max', 'linear_max',
            'logistic_manual', f'mlp',
            'linear', 'poly2',
            'poly3']
    colors = [
            # 'green',
            # 'purple',
            'lightblue', 'darkslateblue',
            'pink', 'orange', 'purple',
            'yellow', 'brown',
            'darkgreen', 'orange',
            'teal', 'pink',
            'red']
    labels = [
            # 'MLP',
            # 'MLP+GNN',
            'Max Ent Edit', 'Max Ent Edit Zero',
            'Zero', 'Log', 'Logistic',
            'Log Max', 'Linear Max',
            'Logistic Manual', 'MLP+PCA',
            'Linear', 'Poly2',
            'Poly3']

    for method, color, label in zip(methods, colors, labels):
        if label not in {'Linear', 'Poly3', 'Max Ent Edit', 'Logistic Manual'}:
            continue
        idir = osp.join(max_ent_dir, f'samples/sample{sample}_{method}')
        ifile = osp.join(idir,'y.npy')
        if osp.exists(ifile):
            y = np.load(ifile)
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            ax.plot(meanDist, label = label, color = color)
            m = len(y)
            mse = mean_squared_error(meanDist_gt[:m], meanDist)
            print(f'\t{label} MSE:', np.round(mse, 8))

            if params:
                # find config
                config = None
                config_file = osp.join(idir, 'config.json')
                if osp.exists(config_file):
                    with open(config_file) as f:
                        config = json.load(f)
                else:
                    # TODO this doesn't work
                    # look for last iteration
                    max_it = -1
                    for f in os.listdir(osp.join(max_ent_dir, f'samples/sample{file}')):
                        if f.startswith('iteration'):
                            it = int(f[9:])
                            if it > max_it:
                                max_it = it
                    config_file = osp.join(dir, file, f'iteration{it}', 'config.json')
                    if osp.exists(config_file):
                        with open(config_file) as f:
                            config = json.load(f)

                diag_chis_step = calculate_diag_chi_step(config)
                ax2.plot(diag_chis_step, ls = '--', label = 'Parameters', color = color)


    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)

    if params:
        ax.set_ylim(None, 50)
        ax.legend(loc='upper left')
        ax2.set_ylabel('Diagonal Parameter', fontsize = 16)
        ax2.legend(loc='upper right')
    else:
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(osp.join(max_ent_dir, 'fitting2/meanDist_edit.png'))
    plt.close()

def find_params_for_synthetic_data(k_arr):
    dataset = 'dataset_11_14_22'
    data_dir = osp.join('/home/erschultz', dataset)

    linear_popt_list = []
    logistic_popt_list = []
    chi_ii_list = []
    chi_ij_list = []
    lmbda_list = []
    f_list = []
    for k in k_arr:
        for sample in range(2201, 2215):
            dir = osp.join(data_dir, f'samples/sample{sample}')
            max_ent_dir = osp.join(dir, f'PCA-normalize-E/k{k}/replicate1')
            if not osp.exists(max_ent_dir):
                continue
            fitting_dir = osp.join(max_ent_dir, 'fitting')
            edit_dir = osp.join(max_ent_dir, 'samples')

            # # get diagonal params
            # linear_popt = np.loadtxt(osp.join(fitting_dir, 'linear_popt.txt'))
            # linear_popt_list.append(linear_popt)
            # logistic_popt = np.loadtxt(osp.join(fitting_dir, 'logistic_manual_popt.txt'))
            # logistic_popt_list.append(logistic_popt)
            #
            # # get lambda from sequences
            # x = np.load(osp.join(max_ent_dir, 'resources/x.npy'))
            # m, k = x.shape
            # for i in range(k):
            #     f, lmbda = Tester.infer_lambda(x[:,i])
            #     f_list.append(f)
            #     lmbda_list.append(lmbda)

            # get plaid params
            chi = np.loadtxt(osp.join(max_ent_dir, 'chis.txt'))[-1]
            chi = triu_to_full(chi)
            for i in range(k):
                for j in range(i+1):
                    chi_ij = chi[i,j]
                    if i == j:
                        chi_ii_list.append(chi_ij)
                    else:
                        chi_ij_list.append(chi_ij)

    linear_popt_arr = np.array(linear_popt_list)
    logistic_popt_arr = np.array(logistic_popt_list)


    # plot plaid parameters
    title = []
    dist = Ecdf()
    for arr, color, label in zip([chi_ii_list, chi_ij_list], ['blue', 'red'], ['ii', 'ij']):
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = 50, alpha = 0.5,
                                    color = color, label = label)
        bin_width = bins[1] - bins[0]
        params = dist.fit(arr)
        dist.save(osp.join(data_dir, f'k{"-".join([str(k) for k in k_arr])}_chi_{label}.json'))
        # print('dist params', params)
        # y = dist.pdf(bins, *params) * bin_width
        # params = [np.round(p, 3) for p in params]
        # title.append(f'{label}, {params}')
        # plt.plot(bins, y, color = color, ls = '--')
    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel('plaid parameter', fontsize=16)
    plt.xlim(-20, 20)
    plt.title('\n'.join(title))
    plt.savefig(osp.join(data_dir, f'k{"-".join([str(k) for k in k_arr])}_chi.png'))
    plt.close()

    # plot plaid parameters with one dist
    # title = []
    # dist = skewnorm
    # chi_ii_list.extend(chi_ij_list)
    # arr = chi_ii_list
    # color = 'red'
    # label = 'ii and ij'
    # n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
    #                             bins = 50, alpha = 0.5, label = label)
    # bin_width = bins[1] - bins[0]
    # params = dist.fit(arr)
    # print('dist params', params)
    # y = dist.pdf(bins, *params) * bin_width
    # params = [np.round(p, 3) for p in params]
    # title.append(f'{label}, {params}')
    # plt.plot(bins, y, color = 'k', ls = '--')
    # plt.ylabel('probability', fontsize=16)
    # plt.xlabel(r'plaid parameter, $\chi_{ij}$', fontsize=16)
    # plt.xlim(-10, 10)
    # plt.title('\n'.join(title))
    # plt.savefig(osp.join(data_dir, f'k{"-".join([str(k) for k in k_arr])}_all_chi.png'))
    # plt.close()

    # arrs = [f_list, lmbda_list, linear_popt_arr[:, 0], linear_popt_arr[:, 1],
    #         logistic_popt_arr[:, 0], logistic_popt_arr[:, 1], logistic_popt_arr[:, 2]]
    # labels = ['f', 'lambda', 'diag_linear_slope', 'diag_linear_intercept',
    #         'diag_logistic_max', 'diag_logistic_slope', 'diag_logistic_midpoint']
    # dist = skewnorm
    # for arr, label in zip(arrs, labels):
    #     if np.min(arr) < 0.001 * np.median(arr):
    #         print(label, np.min(arr))
    #         arr = np.delete(arr, np.argmin(arr), axis = None)
    #         print(arr)
    #     n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
    #                                 bins = 20, color = 'blue', alpha = 0.5)
    #     bin_width = bins[1] - bins[0]
    #     params = dist.fit(arr)
    #     y = dist.pdf(bins, *params) * bin_width
    #     plt.plot(bins, y, ls = '--', color = 'blue')
    #
    #     plt.ylabel('probability', fontsize=16)
    #     plt.xlabel(label, fontsize=16)
    #     params = [np.round(p, 3) for p in params]
    #     plt.title(f'{label}, {params}')
    #     plt.savefig(osp.join(data_dir, f'k{"-".join(k_arr)}_{label}.png'))
    #     plt.close()

def test3():
    # distribution of max ent s_ij after shuffle
    dataset = 'dataset_11_14_22'
    data_dir = osp.join('/home/erschultz', dataset)
    sample=2217
    k=8
    dir = osp.join(data_dir, f'samples/sample{sample}')
    max_ent_dir = osp.join(dir, f'PCA-normalize-E/k{k}/replicate1')
    shuffle_dir = osp.join(max_ent_dir, f'samples/sample{sample}_shuffle_chi')

    s = np.load(osp.join(max_ent_dir, 's.npy'))
    s_shuffle = np.load(osp.join(shuffle_dir, 's.npy'))
    s_shuffle = (s_shuffle + s_shuffle.T)/2
    plot_matrix(s_shuffle, osp.join(shuffle_dir, 's.png'), title = r'$S$', cmap='blue-red')

    # plot plaid parameters
    arr = s[np.triu_indices(len(s))].flatten()
    n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                bins = 50, alpha = 0.5, label = 'original')
    arr = s_shuffle[np.triu_indices(len(s))].flatten()
    n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                bins = 50, alpha = 0.5, label = 'shuffle')

    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(r'$L_{ij}$', fontsize=16)
    plt.xlim(-20, 20)
    plt.savefig(osp.join(data_dir, f's_vs_shuffle_dist.png'))
    plt.close()


def maxent_dist(plot=True):
    # distribution of max ent params
    dataset = 'dataset_01_26_23'
    if dataset == 'dataset_11_14_22':
        samples = range(2201, 2222)
    elif dataset == 'dataset_01_26_23':
        samples = range(201, 258)

    data_dir = osp.join('/home/erschultz', dataset)

    s_list = []
    sd_list = []
    chi_ij_list = []
    chi_list = []
    k = 4
    for sample in samples:
        dir = osp.join(data_dir, f'samples/sample{sample}')
        max_ent_dir = osp.join(dir, f'PCA-normalize-E/k{k}/replicate1')
        if not osp.exists(max_ent_dir):
            continue
        fitting_dir = osp.join(max_ent_dir, 'fitting')
        edit_dir = osp.join(max_ent_dir, 'samples')

        s = np.load(osp.join(max_ent_dir, 's.npy'))
        with open(osp.join(max_ent_dir, 'iteration13/config.json')) as f:
            config = json.load(f)
        d = calculate_D(calculate_diag_chi_step(config))
        sd = calculate_SD(s,d)

        m = len(s)
        s_list.append(s[np.triu_indices(m)])
        sd_list.append(sd[np.triu_indices(m)])

        # get plaid params
        chi = np.loadtxt(osp.join(max_ent_dir, 'chis.txt'))[-1]
        chi = triu_to_full(chi)
        chi_list.append(chi)
        for i in range(k):
            for j in range(i+1):
                chi_ij = chi[i,j]
                if i == j:
                    pass
                else:
                    chi_ij_list.append(chi_ij)

    if plot:
        # plot plaid chi parameters
        title = []
        dist = norm
        arr = np.array(chi_ij_list).reshape(-1)
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = 50, alpha = 0.5)
        bin_width = bins[1] - bins[0]
        params = dist.fit(arr)
        print('dist params', params)
        y = dist.pdf(bins, *params) * bin_width
        params = [np.round(p, 3) for p in params]
        plt.plot(bins, y, ls = '--')
        plt.legend()
        plt.ylabel('probability', fontsize=16)
        plt.xlabel(r'$\chi_{ij}$', fontsize=16)
        plt.xlim(-20, 20)
        plt.title(f'{params}')
        plt.savefig(osp.join(data_dir, f'max_ent_k{k}_chi_dist.png'))
        plt.close()

        # plot plaid Lij parameters
        # title = []
        # dist = skewnorm
        # arr = np.array(s_list).reshape(-1)
        # print(arr, np.min(arr), np.max(arr))
        # n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
        #                             bins = 50, alpha = 0.5)
        # bin_width = bins[1] - bins[0]
        # params = dist.fit(arr)
        # print('dist params', params)
        # y = dist.pdf(bins, *params) * bin_width
        # params = [np.round(p, 3) for p in params]
        # plt.plot(bins, y, ls = '--')
        # plt.legend()
        # plt.ylabel('probability', fontsize=16)
        # plt.xlabel(r'$L_{ij}$', fontsize=16)
        # plt.xlim(-20, 20)
        # plt.title(f'{params}')
        # plt.savefig(osp.join(data_dir, f'max_ent_k{k}_L_dist.png'))
        # plt.close()

        # plaid per chi
        bin_width = 1
        cmap = matplotlib.cm.get_cmap('tab10')
        ind = np.arange(k*(k-1)/2 + k) % cmap.N
        colors = cmap(ind.astype(int))
        fig, ax = plt.subplots(k, k, sharey = True, sharex = True)
        row = 0
        col = 0
        c = 0
        print(colors)
        for i in range(k):
            for j in range(i,k):
                print(c, colors[c])
                data = []
                for chi in chi_list:
                    data.append(chi[i,j])

                dist = norm
                arr = np.array(data).reshape(-1)
                bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
                n, bins, patches = ax[i][j].hist(arr, weights = np.ones_like(arr) / len(arr),
                                            bins = bins, alpha = 0.5, color = colors[c])

                params = dist.fit(arr)
                print('dist params', params)
                y = dist.pdf(bins, *params) * bin_width
                params = np.round(params, 2)
                ax[i][j].plot(bins, y, ls = '--', color = 'k')
                ax[i][j].set_title(r'$\mu$='+f'{params[0]} '+r'$\sigma$='+f'{params[1]}')
                c += 1

        fig.supylabel('probability', fontsize=16)
        plt.xlim(-20, 20)
        plt.tight_layout()
        plt.savefig(osp.join(data_dir, f'max_ent_k{k}_chi_ij_dist.png'))
        plt.close()

        # plaid per pc
        bin_width = 1
        cmap = matplotlib.cm.get_cmap('tab10')
        ind = np.arange(k) % cmap.N
        colors = cmap(ind)
        cols = math.ceil(k/2)
        fig, ax = plt.subplots(2, cols, sharey = True, sharex = True)
        row = 0
        col = 0
        for i in range(k):
            print(i)
            data = []
            for chi in chi_list:
                data.extend(chi[i,:])

            dist = norm
            arr = np.array(data).reshape(-1)
            bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
            n, bins, patches = ax[row][col].hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = bins, alpha = 0.5, color = colors[i])

            params = dist.fit(arr)
            print('dist params', params)
            y = dist.pdf(bins, *params) * bin_width
            params = np.round(params, 3)
            ax[row][col].plot(bins, y, ls = '--', color = colors[i])
            ax[row][col].set_title(f'PC{i}\n'+r'$\mu$='+f'{params[0]} '+r'$\sigma$='+f'{params[1]}')

            col += 1
            if col == cols:
                row += 1
                col = 0

        fig.supylabel('probability', fontsize=16)
        fig.supxlabel(r'$chi_{ij}$', fontsize=16)
        plt.xlim(-20, 20)
        plt.tight_layout()
        plt.savefig(osp.join(data_dir, f'max_ent_k{k}_chi_pc_dist.png'))
        plt.close()


        # plot net energy parameters
        # title = []
        # dist = skewnorm
        # arr = np.array(sd_list).reshape(-1)
        # print(arr, np.min(arr), np.max(arr))
        # n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
        #                             bins = 50, alpha = 0.5)
        # bin_width = bins[1] - bins[0]
        # params = dist.fit(arr)
        # print('dist params', params)
        # y = dist.pdf(bins, *params) * bin_width
        # params = [np.round(p, 3) for p in params]
        # plt.plot(bins, y, ls = '--')
        # plt.legend()
        # plt.ylabel('probability', fontsize=16)
        # plt.xlabel(r'$S_{ij}$', fontsize=16)
        # plt.xlim(-20, 20)
        # plt.title(f'{params}')
        # plt.savefig(osp.join(data_dir, f'max_ent_k{k}_S_dist.png'))
        # plt.close()

    return s_list, chi_ij_list

def simulated_dist(dataset, plot=True):
    # simulated data s_ij distribution
    if dataset == 'dataset_12_20_22':
        samples = [324, 981, 1936, 2834, 3464]
    elif dataset == 'dataset_11_21_22':
        samples = [410, 653, 1462, 1801, 2290]
    else:
        samples = range(1, 11)
    data_dir = osp.join('/home/erschultz', dataset)

    chi_ii_list = []
    chi_ij_list = []
    s_list = []
    sd_list = []
    for sample in samples:
        dir = osp.join(data_dir, f'samples/sample{sample}')
        s = np.load(osp.join(dir, 's.npy'))
        chi = np.load(osp.join(dir, 'chis.npy'))
        diag_chis = np.load(osp.join(dir, 'diag_chis_continuous.npy'))
        d = calculate_D(diag_chis)
        sd = calculate_SD(s, d)

        k = len(chi)
        for i in range(k):
            for j in range(i+1):
                chi_ij = chi[i,j]
                if i == j:
                    chi_ii_list.append(chi_ij)
                else:
                    chi_ij_list.append(chi_ij)

        m = len(s)
        s_list.append(s[np.triu_indices(m)])
        sd_list.append(sd[np.triu_indices(m)])

    if plot:
        # plot plaid L_ij parameters
        title = []
        dist = skewnorm
        arr = np.array(s_list).reshape(-1)
        print(arr, np.min(arr), np.max(arr))
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = 50, alpha = 0.5)
        bin_width = bins[1] - bins[0]
        params = dist.fit(arr)
        print('dist params', params)
        y = dist.pdf(bins, *params) * bin_width
        params = [np.round(p, 3) for p in params]
        plt.plot(bins, y, ls = '--')
        plt.legend()
        plt.ylabel('probability', fontsize=16)
        plt.xlabel(r'$L_{ij}$', fontsize=16)
        plt.xlim(-20, 20)
        plt.title(f'{params}')
        plt.savefig(osp.join(data_dir, f'L_dist.png'))
        plt.close()

        # plot net S_ij parameters
        title = []
        dist = skewnorm
        arr = np.array(sd_list).reshape(-1)
        print(arr, np.min(arr), np.max(arr))
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = 50, alpha = 0.5)
        bin_width = bins[1] - bins[0]
        params = dist.fit(arr)
        print('dist params', params)
        y = dist.pdf(bins, *params) * bin_width
        params = [np.round(p, 3) for p in params]
        plt.plot(bins, y, ls = '--')
        plt.legend()
        plt.ylabel('probability', fontsize=16)
        plt.xlabel(r'$S_{ij}$', fontsize=16)
        plt.xlim(-20, 20)
        plt.title(f'{params}')
        plt.savefig(osp.join(data_dir, f'S_dist.png'))
        plt.close()

        # plot plaid chi parameters
        title = []
        dist = Ecdf()
        for arr, color, label in zip([chi_ii_list, chi_ij_list], ['blue', 'red'], ['ii', 'ij']):
            n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = 50, alpha = 0.5,
                                        color = color, label = label)
            bin_width = bins[1] - bins[0]
            params = dist.fit(arr)

        plt.legend()
        plt.ylabel('probability', fontsize=16)
        plt.xlabel('plaid parameter', fontsize=16)
        plt.xlim(-20, 20)
        plt.title('\n'.join(title))
        plt.savefig(osp.join(data_dir, f'chi_dist.png'))
        plt.close()

    return s_list, chi_ij_list

def compare_maxent_simulation():
    data_dir = '/home/erschultz/dataset_11_14_22'

    s_list = []
    chi_list = []
    l_list = []
    s_max_ent, chi_max_ent = maxent_dist(False)
    s_list.append(s_max_ent)
    chi_list.append(chi_max_ent)
    l_list.append('Max Ent')

    # s_list.append(simulated_dist('dataset_11_21_22', False))
    # l_list.append('Sim Markov')

    # s_sim, chi_sim = simulated_dist('dataset_12_20_22', False)
    # s_list.append(s_sim)
    # chi_list.append(chi_sim)
    # l_list.append('Sim PCs')

    # s_sim, chi_sim = simulated_dist('dataset_01_23_23', False)
    # s_list.append(s_sim)
    # chi_list.append(chi_sim)
    # l_list.append('Sim PCs + chi ecdf')

    s_sim, chi_sim = simulated_dist('dataset_01_27_23', False)
    s_list.append(s_sim)
    chi_list.append(chi_sim)
    l_list.append('Sim PCs + chi multi gauss')

    # plot plaid L_ij parameters
    bin_width = 1
    for arr, label in zip(s_list, l_list):
        arr = np.array(arr).reshape(-1)
        print(arr)
        print(np.min(arr), np.max(arr))
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins=range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width),
                                    alpha = 0.5, label = label)

    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(r'$L_{ij}$', fontsize=16)
    plt.xlim(-20, 20)
    plt.savefig(osp.join(data_dir, 'L_dist_comparison.png'))
    plt.close()

    # plot plaid chi parameters
    bin_width = 1
    for arr, label in zip(chi_list, l_list):
        arr = np.array(arr).reshape(-1)
        print(np.min(arr), np.max(arr))
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins=range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width),
                                    alpha = 0.5, label = label)
    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel(r'$\chi_{ij}$', fontsize=16)
    plt.xlim(-20, 20)
    plt.savefig(osp.join(data_dir, 'chi_dist_comparison.png'))
    plt.close()

def fit_multivariate_gaussian():
    dataset = 'dataset_01_26_23'
    if dataset == 'dataset_11_14_22':
        samples = range(2201, 2222)
    elif dataset == 'dataset_01_26_23':
        samples = range(201, 225)

    data_dir = osp.join('/home/erschultz', dataset)

    chi_list = []
    k=4
    for sample in samples:
        dir = osp.join(data_dir, f'samples/sample{sample}')
        max_ent_dir = osp.join(dir, f'PCA-normalize-E/k{k}/replicate1')
        if not osp.exists(max_ent_dir):
            continue

        # get plaid params
        chi = np.loadtxt(osp.join(max_ent_dir, 'chis.txt'))[-1]
        chi_list.append(chi)

    chi_arr = np.array(chi_list)
    print(chi_arr.shape)
    mean = np.mean(chi_arr, axis = 0)
    cov = np.cov(chi_arr, rowvar = 0)
    with open(f'/home/erschultz/{dataset}/k4_chi.pickle', 'wb') as f:
        dict = {'mean':mean, 'cov':cov}
        pickle.dump(dict, f)
    dist = multivariate_normal
    hat = dist.rvs(mean, cov)
    print(hat)


if __name__ == '__main__':
    # for i in [2217]:
        # modify_maxent_diag_chi(i, k = 6)
        # modify_maxent_diag_chi(i, k = 8, edit = False)
        # plot_modified_max_ent(i, k = 8)
    # find_params_for_synthetic_data([8])
    maxent_dist(True)
    # simulated_dist('dataset_01_27_23')
    # test_ecdf()
    # test3()
    # compare_maxent_simulation()
    # fit_multivariate_gaussian()
