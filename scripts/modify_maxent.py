import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from get_params import Tester
from scipy.optimize import curve_fit
from scipy.stats import gamma, norm, skewnorm, weibull_max, weibull_min
from seq2contact import (DiagonalPreprocessing, calculate_diag_chi_step,
                         triu_to_full)
from sklearn.metrics import mean_squared_error


def modify_maxent_diag_chi(sample, k = 8):
    print(f'sample{sample}, k{k}')
    # try different modifications to diag chis learned by max ent
    dataset = 'dataset_11_14_22'
    dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    max_ent_dir = osp.join(dir, f'PCA_split-binarizeMean-E/k{k}/replicate1')
    odir = osp.join(max_ent_dir, 'fitting')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    diag_chis = np.loadtxt(osp.join(max_ent_dir, 'chis_diag.txt'))[-1]

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

    log_fit = curve_fit_helper(log_curve, x[:64], diag_chi_step[:64], 'log', odir)
    log_max_fit = curve_fit_helper(log_max_curve, x[:64], diag_chi_step[:64],
                                    'log_max', odir)
    logistic_fit = curve_fit_helper(logistic_curve, x[:m], diag_chi_step,
                                    'logistic', odir, [10, 1, 100])
    poly2_fit = curve_fit_helper(poly2_curve, x[:m], diag_chi_step,
                                    'poly2', odir, [1, 1, 1])
    poly3_fit = curve_fit_helper(poly3_curve, x[:m], diag_chi_step,
                                    'poly3', odir, [1, 1, 1, 1])

    try:
        max = np.mean(diag_chi_step[-120:])
        init = [1, 1]
        popt, pcov = curve_fit(logistic_manual_curve, (x[:m], np.ones(m)*max), diag_chi_step, p0 = init, maxfev = 2000)
        popt = np.concatenate((np.atleast_1d(max), popt))
        print('Logistic manual popt', popt)
        logistic_fit_manual = logistic_curve(x, *popt)
        np.savetxt(osp.join(odir, 'logistic_manual_fit.txt'), logistic_fit_manual)
        np.savetxt(osp.join(odir, 'logistic_manual_popt.txt'), popt)
    except RuntimeError as e:
        logistic_fit_manual = None
        print('Logistic manual:', e)



    linear_max_fit = curve_fit_helper(linear_max_curve, x[:m], diag_chi_step,
                                    'linear_max', odir)
    linear_fit = curve_fit_helper(linear_curve, x[:m], diag_chi_step,
                                    'linear', odir)

    for log in [True, False]:
        plt.plot(diag_chi_step, label = 'Max Ent Edit', color = 'lightblue')
        if log_fit is not None:
            plt.plot(log_fit, label = 'log', color = 'orange')
        # if log_max_fit is not None:
        #     plt.plot(log_max_fit, label = 'log_max', color = 'yellow')
        if logistic_fit is not None:
            plt.plot(logistic_fit, label = 'logistic', color = 'purple')
        if logistic_fit_manual is not None:
            plt.plot(logistic_fit_manual, label = 'logistic_manual', color = 'maroon')
        # if linear_max_fit is not None:
        #     plt.plot(linear_max_fit, label = 'linear_max', color = 'brown')
        if linear_fit is not None:
            plt.plot(linear_fit, label = 'linear', color = 'teal')
        # if poly2_fit is not None:
        #     plt.plot(poly2_fit, label = 'poly2', color = 'pink')
        # if poly3_fit is not None:
        #     plt.plot(poly3_fit, label = 'poly3', color = 'red')
        plt.legend()
        plt.ylim(None, 2 * np.max(diag_chi_step))
        if log:
            plt.xscale('log')
            plt.savefig(osp.join(odir, 'meanDist_fit_log.png'))
        else:
            plt.savefig(osp.join(odir, 'meanDist_fit.png'))
        plt.close()

def curve_fit_helper(fn, x, y, label, odir, init = [1,1]):
    try:
        init = [1, 1]
        popt, pcov = curve_fit(log_curve, x, y, p0 = init, maxfev = 2000)
        print(f'\t{label} popt', popt)
        fit = log_curve(x, *popt)
        np.savetxt(osp.join(odir, f'{label}_fit.txt'), fit)
        np.savetxt(osp.join(odir, f'{label}_popt.txt'), popt)
    except RuntimeError as e:
        fit = None
        print(f'{label}:', e)

    return fit

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
    max_ent_dir = osp.join(dir, f'PCA_split-binarizeMean-E/k{k}/replicate1')
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
    plt.savefig(osp.join(max_ent_dir, 'fitting/meanDist_edit.png'))
    plt.close()

def find_params_for_synthetic_data(k_arr):
    dataset = 'dataset_11_14_22'
    data_dir = osp.join('/home/erschultz', dataset)
    k_arr = [str(k) for k in k_arr]

    linear_popt_list = []
    logistic_popt_list = []
    chi_ii_list = []
    chi_ij_list = []
    lmbda_list = []
    f_list = []
    for k in k_arr:
        for sample in [1, 2, 5, 6, 9, 10, 13, 14, 16, 18]:
            dir = osp.join(data_dir, f'samples/sample{sample}')
            max_ent_dir = osp.join(dir, f'PCA_split-binarizeMean-E/k{k}/replicate1')
            fitting_dir = osp.join(max_ent_dir, 'fitting')
            edit_dir = osp.join(max_ent_dir, 'samples')

            # get diagonal params
            linear_popt = np.loadtxt(osp.join(fitting_dir, 'linear_popt.txt'))
            linear_popt_list.append(linear_popt)
            logistic_popt = np.loadtxt(osp.join(fitting_dir, 'logistic_manual_popt.txt'))
            logistic_popt_list.append(logistic_popt)

            # get lambda from sequences
            x = np.load(osp.join(max_ent_dir, 'resources/x.npy'))
            m, k = x.shape
            for i in range(k):
                f, lmbda = Tester.infer_lambda(x[:,i])
                f_list.append(f)
                lmbda_list.append(lmbda)

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
    dist = skewnorm
    for arr, color, label in zip([chi_ii_list, chi_ij_list], ['blue', 'red'], ['ii', 'ij']):
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = 25, alpha = 0.5,
                                    color = color, label = label)
        bin_width = bins[1] - bins[0]
        params = dist.fit(arr)
        print('dist params', params)
        y = dist.pdf(bins, *params) * bin_width
        params = [np.round(p, 3) for p in params]
        title.append(f'{label}, {params}')
        plt.plot(bins, y, color = color, ls = '--')
    plt.legend()
    plt.ylabel('probability', fontsize=16)
    plt.xlabel('plaid parameter', fontsize=16)
    plt.title('\n'.join(title))
    plt.savefig(osp.join(data_dir, f'k{"-".join(k_arr)}_chi.png'))
    plt.close()

    arrs = [f_list, lmbda_list, linear_popt_arr[:, 0], linear_popt_arr[:, 1],
            logistic_popt_arr[:, 0], logistic_popt_arr[:, 1], logistic_popt_arr[:, 2]]
    labels = ['f', 'lambda', 'diag_slope', 'diag_intercept',
            'diag_max', 'diag_slope', 'diag_midpoint']
    dist = skewnorm
    for arr, label in zip(arrs, labels):
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = 20, color = 'blue', alpha = 0.5)
        bin_width = bins[1] - bins[0]
        params = dist.fit(arr)
        y = dist.pdf(bins, *params) * bin_width
        plt.plot(bins, y, ls = '--', color = 'blue')

        plt.ylabel('probability', fontsize=16)
        plt.xlabel(label, fontsize=16)
        params = [np.round(p, 3) for p in params]
        plt.title(f'{label}, {params}')
        plt.savefig(osp.join(data_dir, f'k{"-".join(k_arr)}_{label}.png'))
        plt.close()


if __name__ == '__main__':
    for i in [1, 2, 5, 6, 9, 10, 13, 14, 16, 18]:
        #  1, 2, 5, 6, 9, 10, 13, 14, 16, 18
        # 13 is a meh fit
        modify_maxent_diag_chi(i, k = 8)
        # plot_modified_max_ent(i, k = 8)
    # find_params_for_synthetic_data([8])
