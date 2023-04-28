import json
import math
import os
import os.path as osp
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from ECDF import Ecdf
from MultivariateSkewNormal import multivariate_skewnorm
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import (beta, gamma, laplace, multivariate_normal, norm,
                         skewnorm, weibull_max, weibull_min)
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz/TICG-chromatin')
from scripts.get_params import Tester

sys.path.append('/home/erschultz/TICG-chromatin/pylib')
from utils.energy_utils import (calculate_D, calculate_diag_chi_step,
                                calculate_L, calculate_S)
from utils.plotting_utils import plot_matrix

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_L)
from sequences_to_contact_maps.scripts.plotting_utils import \
    plot_seq_continuous
from sequences_to_contact_maps.scripts.utils import (DiagonalPreprocessing,
                                                     pearson_round,
                                                     triu_to_full)

LETTERS = 'ABCDEFGHIJKLMN'

def get_samples(dataset):
    experimental = False
    if dataset == 'dataset_11_14_22':
        samples = range(2201, 2214)
        experimental = True
    elif dataset in {'dataset_01_26_23', 'dataset_02_04_23', 'dataset_02_21_23'}:
        samples = range(201, 283)
        # samples = range(284, 293)
        # samples = range(201, 210)
        experimental = True
    elif dataset in {'dataset_12_20_22', 'dataset_02_13_23', 'dataset_02_06_23',
                    'dataset_02_22_23'}:
        samples = [324, 981, 1936, 2834, 3464]
    elif dataset == 'dataset_11_21_22':
        samples = [1, 2, 3, 410, 653, 1462, 1801, 2290]
    elif dataset.startswith('dataset_01_27_23'):
        samples = range(1, 16)
    elif dataset in {'dataset_04_09_23', 'dataset_04_10_23',}:
        samples = range(1001, 1028)
    elif dataset == 'dataset_04_06_23':
        samples = range(1001, 1286)
    elif dataset == 'dataset_04_07_23':
        samples = range(1021, 1027)
    else:
        samples = [1, 2, 3, 4, 5, 324, 981, 1936, 2834, 3464]

    return samples, experimental

def modify_plaid_chis(dataset, k):
    samples, _ = get_samples(dataset)
    for sample in samples:
        dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        max_ent_dir = osp.join(dir, f'optimize_grid_b_140_phi_0.06-max_ent')
        chis = np.loadtxt(osp.join(max_ent_dir, 'chis.txt'))
        chis = triu_to_full(chis)
        plot_matrix(chis, osp.join(max_ent_dir, 'chis.png'), cmap = 'blue-red')

        # negative version of chi
        chis_neg = -1 * chis
        plot_matrix(chis_neg, osp.join(max_ent_dir, 'chis_neg.png'), cmap = 'blue-red')
        np.save(osp.join(max_ent_dir, 'chis_neg.npy'), chis_neg)

        # negative version of chi_ij
        chis_neg = -1 * chis
        np.fill_diagonal(chis_neg, np.diagonal(chis))
        plot_matrix(chis_neg, osp.join(max_ent_dir, 'chis_neg_v2.png'), cmap = 'blue-red')
        np.save(osp.join(max_ent_dir, 'chis_neg_v2.npy'), chis_neg)

        # shuffle chis
        chis_shuffle = np.copy(chis)
        diag = np.copy(np.diagonal(chis))
        tri = np.copy(chis[np.triu_indices(len(chis), 1)])
        np.random.shuffle(diag)
        np.fill_diagonal(chis_shuffle, diag)
        np.random.shuffle(tri)
        chis_shuffle[np.triu_indices(len(chis), 1)] = tri
        chis_shuffle = np.triu(chis_shuffle, 1) + np.triu(chis_shuffle).T
        np.save(osp.join(max_ent_dir, 'chis_shuffle.npy'), chis_shuffle)
        plot_matrix(chis_shuffle, osp.join(max_ent_dir, 'chis_shuffle.png'), cmap = 'blue-red')

        # make diagonal by zero out chi_ij
        chis_zeroed = np.zeros_like(chis)
        np.fill_diagonal(chis_zeroed, np.diagonal(chis))
        np.save(osp.join(max_ent_dir, 'chis_zero.npy'), chis_zeroed)
        plot_matrix(chis_zeroed, osp.join(max_ent_dir, 'chis_zero.png'), cmap = 'blue-red')

        # shuffle seq
        x = np.load(osp.join(max_ent_dir, 'iteration0/x.npy'))
        if x.shape[1] > x.shape[0]:
            x = x.T
        x_shuffle = np.copy(x)
        np.random.shuffle(x_shuffle.T)
        np.save(osp.join(max_ent_dir, 'resources/x_shuffle.npy'), x_shuffle)
        np.save(osp.join(max_ent_dir, 'resources/x.npy'), x)


        # compute EIG(L)
        L = calculate_L(x, chis)
        w, V = np.linalg.eig(L)
        x_eig = V[:,:k]

        assert np.sum((np.isreal(x_eig)))
        x_eig = np.real(x_eig)

        chis_eig = np.zeros_like(chis)
        for i, val in enumerate(w[:k]):
            assert np.isreal(val)
            chis_eig[i,i] = np.real(val)

        np.save(osp.join(max_ent_dir, 'resources/x_eig.npy'), x_eig)
        plot_seq_continuous(x_eig, ofile = osp.join(max_ent_dir, 'resources/x_eig.png'))
        np.save(osp.join(max_ent_dir, 'chis_eig.npy'), chis_eig)
        plot_matrix(chis_eig, osp.join(max_ent_dir, 'chis_eig.png'), cmap = 'blue-red')
        L_eig = x_eig @ chis_eig @ x_eig.T
        assert np.allclose(L, L_eig), L - L_eig

        # normalize x_eig
        x_eig_norm = np.zeros_like(x)
        chis_eig_norm = np.zeros_like(chis)
        for i in range(k):
            xi = x_eig[:, i]
            min = np.min(xi)
            max = np.max(xi)
            if max > abs(min):
                val = max
            else:
                val = abs(min)

            # multiply by scale such that val x scale = 1
            scale = 1/val
            x_eig_norm[:,i] = xi * scale

            # multiply by val**2 to counteract
            chis_eig_norm[i,i] = chis_eig[i,i] * val * val

        np.save(osp.join(max_ent_dir, 'resources/x_eig_norm.npy'), x_eig_norm)
        plot_seq_continuous(x_eig_norm, ofile = osp.join(max_ent_dir, 'resources/x_eig_norm.png'))
        np.save(osp.join(max_ent_dir, 'chis_eig_norm.npy'), chis_eig_norm)
        plot_matrix(chis_eig_norm, osp.join(max_ent_dir, 'chis_eig_norm.png'), cmap = 'blue-red')
        L_eig = x_eig_norm @ chis_eig_norm @ x_eig_norm.T
        assert np.allclose(L, L_eig), L - L_eig


def modify_maxent_diag_chi(dataset, k = 8, edit = True):
    '''
    Inputs:
        k: number of marks
        edit: True to modify maxent result so that is is flat at start
    '''
    samples, _ = get_samples(dataset)
    for sample in samples:
        print(f'sample{sample}, k{k}')
        # try different modifications to diag chis learned by max ent
        dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        max_ent_dir = osp.join(dir, f'optimize_grid_b_140_phi_0.06-max_ent')
        if not osp.exists(max_ent_dir):
            print(f'{max_ent_dir} does not exist')
            continye
        if edit:
            odir = osp.join(max_ent_dir, 'fitting')
        else:
            odir = osp.join(max_ent_dir, 'fitting2')
        if not osp.exists(odir):
            os.mkdir(odir, mode = 0o755)

        diag_chis = np.loadtxt(osp.join(max_ent_dir, 'chis_diag.txt'))
        diag_chis = np.atleast_2d(diag_chis)[-1]

        if edit:
            # make version that is flat at start
            min_val = np.min(diag_chis[:20])
            min_ind = np.argmin(diag_chis[:20])
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
            smoothed_fit = None
            piecewise_linear_fit = curve_fit_helper(Curves.piecewise_linear_curve, x[:m],
                                    diag_chi_step, 'piecewise_linear', odir,
                                    [1, 1, 1, 1, 10], start = 2)
            piecewise_poly2_fit = None
            piecewise_poly3_fit = None
            log_fit = curve_fit_helper(Curves.log_curve, x[:64], diag_chi_step[:64],
                                            'log', odir)
            log_max_fit = curve_fit_helper(Curves.log_max_curve, x[:64], diag_chi_step[:64],
                                            'log_max', odir)
            logistic_fit = curve_fit_helper(Curves.logistic_curve, x[:m], diag_chi_step,
                                            'logistic', odir, [10, 1, 100])
            poly2_fit = curve_fit_helper(Curves.poly2_curve, x[:m], diag_chi_step,
                                            'poly2', odir, [1, 1, 1])
            poly2_log_fit = None
            poly3_fit = curve_fit_helper(Curves.poly3_curve, x[:m], diag_chi_step,
                                            'poly3', odir, [1, 1, 1, 1])
            poly3_log_fit = None
            poly4_log_fit = None
            linear_max_fit = curve_fit_helper(Curves.linear_max_curve, x[:m], diag_chi_step,
                                            'linear_max', odir)
            linear_fit = curve_fit_helper(Curves.linear_curve, x[:m], diag_chi_step,
                                            'linear', odir)
        else:
            smoothed_fit = gaussian_filter(diag_chi_step[2:], sigma = 1)
            smoothed_fit = np.append(diag_chi_step[:2], smoothed_fit)
            log_fit = None
            log_max_fit = None
            logistic_fit = None
            poly2_fit = curve_fit_helper(Curves.poly2_curve, x[:m], diag_chi_step,
                                            'poly2', odir, [1, 1, 1],
                                            start = 2)
            poly2_log_fit = curve_fit_helper(Curves.poly2_curve, np.log(x[:m]), diag_chi_step,
                                            'poly2_log', odir, [1, 1, 1], start = 2)

            poly3_fit = curve_fit_helper(Curves.poly3_curve, x[:m], diag_chi_step,
                                            'poly3', odir, [1, 1, 1, 1],
                                            start = 2)
            poly3_log_fit = curve_fit_helper(Curves.poly3_curve, np.log(x[:m]), diag_chi_step,
                                            'poly3_log', odir, [1, 1, 1, 1], start = 2)
            poly4_log_fit = curve_fit_helper(Curves.poly4_curve, np.log(x[:m]), diag_chi_step,
                                            'poly4_log', odir, [1, 1, 1, 1, 1], start = 2)
            poly6_log_fit = curve_fit_helper(Curves.poly6_curve, np.log(x[:m]), diag_chi_step,
                                            'poly6_log', odir, [1, 1, 1, 1, 1, 1, 1], start = 2)


            piecewise_linear_fit = curve_fit_helper(Curves.piecewise_linear_curve, x[:m],
                                    diag_chi_step, 'piecewise_linear', odir,
                                    [1, 1, 1, 1, 10], start = 2)
            piecewise_poly2_fit = curve_fit_helper(Curves.piecewise_poly2_curve, x[:m],
                                    diag_chi_step, 'piecewise_poly2', odir,
                                    [1, 1, 1, 1, 1, 1, 10], start = 2)
            piecewise_poly3_fit = curve_fit_helper(Curves.piecewise_poly3_curve, x[:m],
                                    diag_chi_step, 'piecewise_poly3', odir,
                                    [1, 1, 1, 1, 1, 1, 1, 1, 20], start = 2)
            linear_max_fit = None
            linear_fit = None


        for log in [True, False]:
            plt.plot(diag_chi_step, label = 'Max Ent Edit', color = 'lightblue')
            if smoothed_fit is not None:
                plt.plot(smoothed_fit, label = 'Gaussian Filter', color = 'lightblue', ls='dashdot')
            if log_fit is not None:
                plt.plot(log_fit, label = 'log', color = 'orange')
            # if log_max_fit is not None:
            #     plt.plot(log_max_fit, label = 'log_max', color = 'yellow')
            if logistic_fit is not None:
                plt.plot(logistic_fit, label = 'logistic', color = 'purple')
            # if linear_max_fit is not None:
            #     plt.plot(linear_max_fit, label = 'linear_max', color = 'brown')
            # if linear_fit is not None:
            #     plt.plot(linear_fit, label = 'linear', color = 'teal')
            # if piecewise_linear_fit is not None:
            #     plt.plot(piecewise_linear_fit, label = 'piecewise_linear', color = 'teal', ls='--')
            # if poly2_fit is not None:
            #     plt.plot(poly2_fit, label = 'poly2', color = 'pink')
            # if piecewise_poly2_fit is not None:
            #     plt.plot(piecewise_poly2_fit, label = 'piecewise_poly2', color = 'pink', ls='--')
            # if poly2_log_fit is not None:
            #     plt.plot(poly2_log_fit, label = 'poly2_log', color = 'pink', ls=':')
            # if poly3_fit is not None:
            #     plt.plot(poly3_fit, label = 'poly3', color = 'red')
            # if poly3_log_fit is not None:
            #     plt.plot(poly3_log_fit, label = 'poly3_log', color = 'red', ls=':')
            # if piecewise_poly3_fit is not None:
            #     plt.plot(piecewise_poly3_fit, label = 'piecewise_poly3', color = 'red', ls='--')
            if poly4_log_fit is not None:
                plt.plot(poly4_log_fit, label = 'poly4_log', color = 'green', ls=':')
            if poly6_log_fit is not None:
                plt.plot(poly6_log_fit, label = 'poly6_log', color = 'darkgreen', ls=':')

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
        fit = fn(x[start:], *popt)
        if start > 0:
            fit = np.append(np.zeros(start), fit)
        np.savetxt(osp.join(odir, f'{label}_fit.txt'), fit)
        np.savetxt(osp.join(odir, f'{label}_popt.txt'), popt)
    except RuntimeError as e:
        fit = None
        print(f'{label}:', e)

    return fit

class Curves():
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

    def poly2_curve(x, A, B, C):
        result = A + B*x + C*x**2
        return result

    def poly3_curve(x, A, B, C, D):
        result = A + B*x + C*x**2 + D*x**3
        return result

    def poly4_curve(x, A, B, C, D, E):
        result = A + B*x + C*x**2 + D*x**3 + E*x**4
        return result

    def poly6_curve(x, A, B, C, D, E, F, G):
        result = A + B*x + C*x**2 + D*x**3 + E*x**4 + F*x**5 + G*x**6
        return result



def plot_modified_max_ent(sample, params = True, k = 8):
    # plot different p(s) curves
    print(f'sample{sample}, k{k}')
    dataset = 'dataset_02_04_23'
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
            'poly3', 'poly4_log',
            'poly6_log']
    colors = [
            # 'green',
            # 'purple',
            'lightblue', 'darkslateblue',
            'pink', 'orange', 'purple',
            'yellow', 'brown',
            'darkgreen', 'orange',
            'teal', 'pink',
            'red', 'green',
            'darkgreen']
    labels = [
            # 'MLP',
            # 'MLP+GNN',
            'Max Ent Edit', 'Max Ent Edit Zero',
            'Zero', 'Log', 'Logistic',
            'Log Max', 'Linear Max',
            'Logistic Manual', 'MLP+PCA',
            'Linear', 'Poly2',
            'Poly3', 'Poly4_log',
            'Poly6_log']

    for method, color, label in zip(methods, colors, labels):
        if label not in {'Linear',  'Poly6_log'}:
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

def test_shuffle():
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

def simple_histogram(arr, xlabel='X', odir=None, ofname=None, dist=skewnorm,
                    label=None, legend_title=''):
    title = []
    if arr is not None:
        arr = np.array(arr).reshape(-1)
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                    bins = 50, alpha = 0.5, label = label)
        bin_width = bins[1] - bins[0]
        params = dist.fit(arr)
        y = dist.pdf(bins, *params) * bin_width
        params = [np.round(p, 3) for p in params]
        print(ofname, params)
        if dist == skewnorm and ofname is not None:
            with open(osp.join(odir, ofname[:-9]+'.pickle'), 'wb') as f:
                dict = {'alpha':params[0], 'mu':params[1], 'sigma':params[2]}
                pickle.dump(dict, f)
        plt.plot(bins, y, ls = '--', color = 'k')
    if not (odir is None or ofname is None):
        if label is not None:
            plt.legend(title = legend_title)
        plt.ylabel('probability', fontsize=16)
        plt.xlabel(xlabel, fontsize=16)
        # plt.xlim(-20, 20)
        if dist == skewnorm and arr is not None:
            plt.title(r'$\alpha=$' + f'{params[0]}\n' + r'$\mu$=' + f'{params[-2]} '+r'$\sigma$='+f'{params[-1]}')
        plt.savefig(osp.join(odir, ofname))
        plt.close()

def simple_scatter(arr_x, arr_y, label_x, label_y, color=None, odir=None, ofname=None):
    X = arr_x
    X = sm.add_constant(X)
    Y = arr_y
    est = sm.OLS(Y, X)
    est = est.fit()
    params = np.round(est.params, 3)
    X = np.linspace(min(arr_x), max(arr_x), 100)
    plt.plot(X, est.predict(sm.add_constant(X)), ls='--', c = 'k')

    plt.title(f'y={params[0]}x+{params[1]}\n'+r'$R^2=$'+f'{np.round(est.rsquared, 2)}')
    if color is None:
        plt.scatter(arr_x, arr_y)
    else:
        plt.scatter(arr_x, arr_y, c = color, cmap = 'RdPu')
    plt.xlabel(label_x, fontsize=16)
    plt.ylabel(label_y, fontsize=16)
    plt.tight_layout()
    if odir is not None and ofname is not None:
        plt.savefig(osp.join(odir, ofname))
    else:
        plt.show()
    plt.close()


def diagonal_dist(dataset, k, plot=True):
    # distribution of diagonal params
    samples, experimental = get_samples(dataset)
    dir = '/project2/depablo/erschultz/'
    if not osp.exists(dir):
        dir = '/home/erschultz'
    data_dir = osp.join(dir, dataset)

    grid_size_arr = grid_dist(dataset, False)
    linear_popt_list = []
    # logistic_popt_list = []
    for sample in samples:
        dir = osp.join(data_dir, f'samples/sample{sample}')
        max_ent_dir = osp.join(dir, f'PCA-normalize-E/k{k}/replicate1')
        if not osp.exists(max_ent_dir):
            continue
        fitting_dir = osp.join(max_ent_dir, 'fitting')

        # get diagonal params
        linear_popt = np.loadtxt(osp.join(fitting_dir, 'linear_popt.txt'))
        linear_popt_list.append(linear_popt)
        # logistic_popt = np.loadtxt(osp.join(fitting_dir, 'logistic_manual_popt.txt'))
        # logistic_popt_list.append(logistic_popt)

    linear_popt_arr = np.array(linear_popt_list)
    # logistic_popt_arr = np.array(logistic_popt_list)

    delete_arr = np.zeros(len(linear_popt_arr), dtype=bool)
    for col in [0,1]:
        arr = linear_popt_arr[:, col]
        # remove outliers
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        l_cutoff = np.percentile(arr, 25) - 1.5 * iqr
        u_cutoff = np.percentile(arr, 75) + 1.5 * iqr
        print(col, iqr, (l_cutoff, u_cutoff))

        tmp_delete_arr = np.logical_or(arr < l_cutoff, arr > u_cutoff)
        delete_arr = np.logical_or(delete_arr, tmp_delete_arr)

    linear_popt_arr = np.delete(linear_popt_arr, delete_arr, axis = 0)
    grid_size_arr = np.delete(grid_size_arr, delete_arr, axis = 0)



    odir = osp.join(data_dir, 'diagonal_param_distributions')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    slope = linear_popt_arr[:, 0]
    intercept = linear_popt_arr[:, 1]
    arrs = [slope, intercept]
            # logistic_popt_arr[:, 0], logistic_popt_arr[:, 1], logistic_popt_arr[:, 2]]
    labels = ['linear_slope', 'linear_intercept']
            # 'diag_logistic_max', 'diag_logistic_slope', 'diag_logistic_midpoint']
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
        plt.title(f'{label}\n' + r'$\mu$=' + f'{params[1]}, ' + r'$\sigma$='
                + f'{params[2]}, ' + r'$\alpha$=' + f'{params[0]}')
        plt.savefig(osp.join(odir, f'max_ent_k{k}_{label}.png'))
        plt.close()

        with open(osp.join(odir, f'k{k}_{label}.pickle'), 'wb') as f:
            dict = {'alpha':params[0], 'mu':params[1], 'sigma':params[2]}
            pickle.dump(dict, f)


    # scatter of slope vs intercept
    simple_scatter(slope, intercept, 'slope', 'intercept', grid_size_arr, odir, 'slope_vs_intercept.png')

    # scatter of slope vs grid_size
    simple_scatter(slope, grid_size_arr, 'slope', 'grid_size', intercept, odir, 'slope_vs_grid_size.png')

    # scatter of slope vs grid_size
    simple_scatter(intercept, grid_size_arr, 'intercept', 'grid_size', slope, odir, 'intercept_vs_grid_size.png')

    # 3d scatter
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter(slope, grid_size_arr, intercept)
    ax.set_xlabel('slope')
    ax.set_ylabel('grid_size')
    ax.set_zlabel('intercept')
    plt.savefig(osp.join(odir, 'slope_vs_intercept_vs_grid_size.png'))
    plt.close()


def seq_dist(dataset, k, plot=True, eig=False, eig_norm=False):
    # distribution of seq params
    samples, experimental = get_samples(dataset)
    dir = '/project2/depablo/erschultz/'
    if not osp.exists(dir):
        dir = '/home/erschultz'
    data_dir = osp.join(dir, dataset)
    if eig:
        odir = osp.join(data_dir, 'seq_param_distributions_eig')
    elif eig_norm:
        odir = osp.join(data_dir, 'seq_param_distributions_eig_norm')
    else:
        odir = osp.join(data_dir, 'seq_param_distributions')

    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    N = len(samples)
    lmbda_arr = np.zeros((N, k))
    f_arr = np.zeros((N, k))
    x_list = []
    for i, sample in enumerate(samples):
        dir = osp.join(data_dir, f'samples/sample{sample}')
        if experimental:
            dir = osp.join(dir, f'PCA-normalize-E/k{k}/replicate1')
        if not osp.exists(dir):
            continue

        # get seq
        if eig:
            x = np.load(osp.join(dir, 'resources/x_eig.npy'))
        elif eig_norm:
            x = np.load(osp.join(dir, 'resources/x_eig_norm.npy'))
        elif experimental:
            x = np.load(osp.join(dir, 'resources/x.npy'))
        else:
            x = np.load(osp.join(dir, 'x.npy'))
        x_list.append(x)

        lmbda_i_list = []
        f_i_list = []
        for j in range(k):
            f, lmbda = Tester.infer_lambda(x[:, j], True)
            if np.isnan(lmbda):
                lmbda = 0
            f_arr[i,j] = f
            lmbda_arr[i,j] = lmbda

    if plot:
        simple_histogram(f_arr, 'f', odir,
                            f'k{k}_f_dist.png', dist = skewnorm)
        simple_histogram(lmbda_arr, r'$\lambda$', odir,
                            f'k{k}_lambda_dist.png', dist = skewnorm)

        # f dist per seq
        cmap = matplotlib.cm.get_cmap('tab10')
        ind = np.arange(k) % cmap.N
        colors = cmap(ind.astype(int))
        dist = skewnorm
        for inp_arr, label in zip([f_arr, lmbda_arr], ['f', 'lambda']):
            fig, ax = plt.subplots(1, k)
            c = 0
            for i in range(k):
                arr = inp_arr[:, i].reshape(-1)
                n, bins, _ = ax[i].hist(arr, weights = np.ones_like(arr) / len(arr),
                                            bins = 30, alpha = 0.5, color = colors[c])
                bin_width = bins[1] - bins[0]

                params = dist.fit(arr)
                y = dist.pdf(bins, *params) * bin_width
                params = np.round(params, 1)
                with open(osp.join(odir, f'k{k}_f_{LETTERS[i]}.pickle'), 'wb') as f:
                    dict = {'alpha':params[0], 'mu':params[1], 'sigma':params[2]}
                    pickle.dump(dict, f)


                ax[i].plot(bins, y, ls = '--', color = 'k')
                ax[i].set_title(r'$\alpha=$' + f'{params[0]}\n' + r'$\mu$=' + f'{params[-2]} '+r'$\sigma$='+f'{params[-1]}')
                c += 1

            fig.supxlabel(f'{label}', fontsize=16)
            fig.supylabel('probability', fontsize=16)
            plt.tight_layout()
            plt.savefig(osp.join(odir, f'k{k}_{label}_per_dist.png'))
            plt.close()


    return x_list

def plaid_dist(dataset, k=None, plot=True, eig=False, eig_norm=False):
    # distribution of plaid params
    samples, experimental = get_samples(dataset)
    dir = '/project2/depablo/erschultz/'
    if not osp.exists(dir):
        dir = '/home/erschultz'
    data_dir = osp.join(dir, dataset)
    if eig:
        odir = osp.join(data_dir, 'plaid_param_distributions_eig')
    elif eig_norm:
        odir = osp.join(data_dir, 'plaid_param_distributions_eig_norm')
    else:
        odir = osp.join(data_dir, 'plaid_param_distributions')
    print(odir)

    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    L_list = []
    D_list = []
    S_list = []
    chi_ij_list = []
    chi_ii_list = []
    chi_list = []
    chi_flat_list = []
    grid_size_arr = grid_dist(dataset, False)
    for sample in samples:
        dir = osp.join(data_dir, f'samples/sample{sample}')
        if experimental:
            dir = osp.join(dir, 'optimize_grid_b_140_phi_0.06-max_ent')
        if not osp.exists(dir):
            continue

        # get L
        L = load_L(dir)
        L = (L+L.T)/2 # ensure L is symmetric

        # get D
        if experimental:
            folder = get_final_max_ent_folder(dir)
            with open(osp.join(folder, 'config.json')) as f:
                config = json.load(f)
            D = calculate_D(calculate_diag_chi_step(config))
        else:
            diag_chis_file = osp.join(dir, 'diag_chis_continuous.npy')
            if osp.exists(diag_chis_file):
                diag_chis = np.load(diag_chis_file)
            else:
                diag_chis = np.load(osp.join(dir, 'diag_chis.npy'))
            D = calculate_D(diag_chis)
        S = calculate_S(L, D)

        m = len(L)
        L_list.append(L[np.triu_indices(m)])
        D_list.append(D[np.triu_indices(m)])
        S_list.append(S[np.triu_indices(m)])

        # get chi
        if eig:
            chi = np.load(osp.join(dir, 'chis_eig.npy'))
        elif eig_norm:
            chi = np.load(osp.join(dir, 'chis_eig_norm.npy'))
        elif experimental:
            chi = np.loadtxt(osp.join(dir, 'chis.txt'))
            chi = np.atleast_2d(chi)[-1]
            chi_flat_list.append(chi)
            chi = triu_to_full(chi)
        else:
            chi_file = osp.join(dir, 'chis.npy')
            if osp.exists(chi_file):
                chi = np.load(chi_file)
                k = len(chi)
            else:
                chi = None

        if chi is not None:
            chi_list.append(chi)
            for i in range(k):
                for j in range(i+1):
                    chi_ij = chi[i,j]
                    if i == j:
                        chi_ii_list.append(chi_ij)
                    else:
                        chi_ij_list.append(chi_ij)

    # multivariate normal
    if chi_flat_list:
        chi_arr = np.array(chi_flat_list)
        mean = np.mean(chi_arr, axis = 0)
        # print(mean)
        cov = np.cov(chi_arr, rowvar = 0)
        with open(osp.join(odir, f'k{k}_chi_multivariate.pickle'), 'wb') as f:
            dict = {'mean':mean, 'cov':cov}
            pickle.dump(dict, f)
        dist = multivariate_normal
        # for _ in range(10):
        #     hat = dist.rvs(mean, cov)
        #     print(hat)

    if plot:
        x_list = seq_dist(dataset, k, plot, eig, eig_norm)
        # plot plaid chi parameters
        if not (eig or eig_norm):
            simple_histogram(chi_ij_list, r'$\chi_{ij}$', odir,
                                f'k{k}_chi_ij_dist.png', dist = laplace)

        # plaid chi_ii parameters
        simple_histogram(chi_ii_list, r'$\chi_{ii}$', odir,
                            f'k{k}_chi_ii_dist.png', dist = skewnorm)

        # plot plaid Lij parameters
        simple_histogram(L_list, r'$L_{ij}$', odir,
                            f'k{k}_L_dist.png')

        # plot net energy parameters
        # simple_histogram(s_list, r'$S_{ij}$', odir,
        #                     f'k{k}_S_dist.png')

        # if not (eig or eig_norm):
        #     # corr(A,B) vs chi_AB
        #     for log in [True, False]:
        #         X = []
        #         Y = []
        #         for i in range(k):
        #             for j in range(i+1,k):
        #                 for s in range(len(samples)):
        #                     chi = chi_list[s]
        #                     chi_ij = chi[i,j]
        #                     if log:
        #                         chi_ij = np.sign(chi_ij) * np.log(np.abs(chi_ij)+1)
        #                     X.append(chi_ij)
        #                     corr = pearson_round(x_list[s][:, i], x_list[s][:, j])
        #                     # if log:
        #                         # corr = np.log(corr + 1)
        #                     Y.append(corr)
        #
        #         est = sm.OLS(Y, X)
        #         est = est.fit()
        #         params = np.round(est.params, 3)
        #         plt.plot(X, est.predict(X), ls='--', c = 'k')
        #         plt.axhline(0, c = 'gray')
        #         plt.axvline(0, c = 'gray')
        #
        #         plt.title(f'y={params[0]}x\n'+r'$R^2=$'+f'{np.round(est.rsquared, 2)}')
        #         plt.scatter(X, Y)
        #         plt.ylabel(r'Pearson($\psi_A$, $\psi_B$)', fontsize=16)
        #         if log:
        #             plt.xlabel(r'sign$(\chi_{AB})$*ln(|$\chi_{AB}$|+1)', fontsize=16)
        #         else:
        #             plt.xlabel(r'$\chi_{AB}$', fontsize=16)
        #         plt.xlim(-5 ,5)
        #         plt.tight_layout()
        #         if log:
        #             plt.savefig(osp.join(odir, f'chi_ij_ln_vs_corr_ij.png'))
        #         else:
        #             plt.savefig(osp.join(odir, f'chi_ij_vs_corr_ij.png'))
        #         plt.close()
        #
        #     # plot chi_ij conditioned on corr(psi_i, psi_j)
        #     X_neg = []
        #     X_pos = []
        #     for i in range(k):
        #         for j in range(i+1,k):
        #             for s in range(len(samples)):
        #                 chi = chi_list[s]
        #                 chi_ij = chi[i,j]
        #                 corr = pearson_round(x_list[s][:, i], x_list[s][:, j])
        #                 if corr > 0:
        #                     X_neg.append(chi_ij)
        #                 else:
        #                     X_pos.append(chi_ij)
        #
        #     simple_histogram(X_neg, dist = skewnorm, label='negative')
        #     simple_histogram(X_pos, r'$\chi_{AB}$', odir,
        #                         f'k{k}_chi_ij_dist_conditioned.png', dist = skewnorm,
        #                         label = 'positive', legend_title = r'Pearson($\psi_A$, $\psi_B$)')
        #
        #     # bivariate distributions
        #     triples = [(2, 3, 'CD'), (1, 2, 'BC'), (1, 3, 'BD')]
        #     X = []
        #     for s in range(len(samples)):
        #         chi = chi_list[s]
        #         X.append(chi[0,1])
        #     fig, ax = plt.subplots(1, 3, sharex = True)
        #     for col, (i, j, label) in enumerate(triples):
        #         Y = []
        #         for s in range(len(samples)):
        #             chi = chi_list[s]
        #             Y.append(chi[i,j])
        #
        #         ax[col].scatter(X, Y)
        #         ax[col].set_xlabel(r'$\chi_{AB}$', fontsize=16)
        #         ax[col].set_ylabel(r'$\chi$'+label, fontsize=16)
        #
        #     plt.tight_layout()
        #     plt.savefig(osp.join(odir, f'chi_AB_bivariates.png'))
        #     plt.close()
        # else:
        #     log = False
        #     X = []
        #     Y_mean = []
        #     Y_f = []
        #     Y_lambda = []
        #     for i in range(k):
        #         for s in range(len(samples)):
        #             chi = chi_list[s]
        #             chi_ij = chi[i,i]
        #             if log:
        #                 chi_ij = np.sign(chi_ij) * np.log(np.abs(chi_ij)+1)
        #             X.append(chi_ij)
        #
        #             Y_mean.append(np.mean(x_list[s][:, i]))
        #             f, lmbda = Tester.infer_lambda(x_list[s][:, i], True)
        #             if np.isnan(lmbda):
        #                 lmbda = 0
        #             Y_f.append(f)
        #             Y_lambda.append(lmbda)
        #
        #     for Y, label in zip([Y_mean, Y_f, Y_lambda], ['mean', 'f', 'lambda']):
        #         print(len(X), len(Y))
        #         est = sm.OLS(Y, X)
        #         est = est.fit()
        #         params = np.round(est.params, 3)
        #         plt.plot(X, est.predict(X), ls='--', c = 'k')
        #         plt.axhline(0, c = 'gray')
        #         plt.axvline(0, c = 'gray')
        #
        #         # plt.title(f'y={params[0]}x\n'+r'$R^2=$'+f'{np.round(est.rsquared, 2)}')
        #         print(len(X), len(Y), X[:2], Y[:2])
        #         plt.scatter(X, Y)
        #         plt.ylabel(label, fontsize=16)
        #         if log:
        #             plt.xlabel(r'sign$(\chi_{ii})$*ln(|$\chi_{ii}$|+1)', fontsize=16)
        #         else:
        #             plt.xlabel(r'$\chi_{ii}$', fontsize=16)
        #         # plt.xlim(-5 ,5)
        #         plt.tight_layout()
        #         if log:
        #             plt.savefig(osp.join(odir, f'k{k}_chi_ii_ln_vs_{label}.png'))
        #         else:
        #             plt.savefig(osp.join(odir, f'k{k}_chi_ii_vs_{label}.png'))
        #         plt.close()
        #

        # plaid per chi
        print("Starting plaid per chi")
        bin_width = 1
        cmap = matplotlib.cm.get_cmap('tab10')
        ind = np.arange(k) % cmap.N
        colors = cmap(ind.astype(int))
        # per chi ii
        rows = math.ceil(k / 4)
        cols = min(4, k)
        fig, ax = plt.subplots(rows, cols)
        if rows == 1:
            ax = [ax]
        c = 0
        row = 0
        col = 0
        for i in range(k):
            print(f'k={i}')
            data = []
            for chi in chi_list:
                data.append(chi[i,i])
            arr = np.array(data).reshape(-1)

            # remove outliers
            iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
            width = 1.5
            l_cutoff = np.percentile(arr, 25) - width * iqr
            u_cutoff = np.percentile(arr, 75) + width * iqr

            delete_arr = np.logical_or(arr < l_cutoff, arr > u_cutoff)
            print(i, iqr, (l_cutoff, u_cutoff))
            arr = np.delete(arr, delete_arr, axis = None)

            dist = skewnorm

            bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
            n, bins, patches = ax[row][col].hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = bins, alpha = 0.5, color = colors[c])

            params = dist.fit(arr)
            y = dist.pdf(bins, *params) * bin_width
            params = np.round(params, 1)
            with open(osp.join(odir, f'k{k}_chi{LETTERS[i]}{LETTERS[i]}.pickle'), 'wb') as f:
                dict = {'alpha':params[0], 'mu':params[1], 'sigma':params[2]}
                pickle.dump(dict, f)


            ax[row][col].plot(bins, y, ls = '--', color = 'k')
            # ax[row][col].set_title(r'$\alpha=$' + f'{params[0]}\n' + r'$\mu$=' + f'{params[-2]} '+r'$\sigma$='+f'{params[-1]}')
            ax[row][col].set_xlabel(rf'$\chi${LETTERS[i]+LETTERS[i]}')

            col += 1
            if col == cols:
                col = 0
                row += 1
            c += 1

        fig.supxlabel(r'$\chi_{ii}$', fontsize=16)
        fig.supylabel('probability', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'k{k}_chi_per_ii_dist.png'))
        plt.close()

        # per chi all
        if not (eig or eig_norm):
            print('Starting per chi all')
            ind = np.arange(k*(k-1)/2 + k) % cmap.N
            colors = cmap(ind.astype(int))

            fig, axes = plt.subplots(k, k, sharey = True, sharex = True)
            row = 0
            col = 0
            c = 0
            for i in range(k):
                for j in range(k):
                    ax = axes[i][j]
                    if j < i:
                        # ax.get_xaxis().set_visible(False)
                        # ax.get_yaxis().set_visible(False)
                        continue

                    data = []
                    for chi in chi_list:
                        data.append(chi[i,j])

                    dist = skewnorm
                    arr = np.array(data).reshape(-1)
                    bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
                    n, bins, patches = ax.hist(arr, weights = np.ones_like(arr) / len(arr),
                                                bins = bins, alpha = 0.5, color = colors[c])

                    params = dist.fit(arr)
                    y = dist.pdf(bins, *params) * bin_width
                    params = np.round(params, 1)
                    pair = LETTERS[i] + LETTERS[j]
                    with open(osp.join(odir, f'k{k}_chi{pair}.pickle'), 'wb') as f:
                        dict = {'alpha':params[0], 'mu':params[1], 'sigma':params[2]}
                        pickle.dump(dict, f)

                    ax.plot(bins, y, ls = '--', color = 'k')
                    # ax.set_title(r'$\alpha=$' + f'{params[0]}\n' + r'$\mu$=' + f'{params[-2]} '+r'$\sigma$='+f'{params[-1]}')
                    c += 1

            fig.supxlabel(r'$\chi_{ij}$', fontsize=16)
            fig.supylabel('Probability', fontsize=16)
            plt.xlim(-20, 20)
            plt.tight_layout()
            plt.savefig(osp.join(odir, f'k{k}_chi_per_dist.png'))
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
            data = []
            for chi in chi_list:
                data.extend(chi[i,:])

            dist = norm
            arr = np.array(data).reshape(-1)
            bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
            n, bins, patches = ax[row][col].hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = bins, alpha = 0.5, color = colors[i])

            params = dist.fit(arr)
            y = dist.pdf(bins, *params) * bin_width
            params = np.round(params, 3)
            ax[row][col].plot(bins, y, ls = '--', color = colors[i])
            ax[row][col].set_title(f'PC{i}\n'+r'$\mu$='+f'{params[-2]} '+r'$\sigma$='+f'{params[-1]}')

            col += 1
            if col == cols:
                row += 1
                col = 0

        fig.supylabel('probability', fontsize=16)
        fig.supxlabel(r'$\chi_{ij}$', fontsize=16)
        # plt.xlim(-20, 20)
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'k{k}_chi_pc_dist.png'))
        plt.close()

        if grid_size_arr is not None:
            # grid_size vs chi_aa
            data = []
            for chi in chi_list:
                data.append(chi[0,0])

            # remove outliers
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            width = 1.5
            l_cutoff = np.percentile(data, 25) - width * iqr
            u_cutoff = np.percentile(data, 75) + width * iqr

            delete_arr = np.logical_or(data < l_cutoff, data > u_cutoff) # ind to delete
            data = np.delete(data, delete_arr, axis = None)
            data2 = np.delete(grid_size_arr, delete_arr, axis = None)
            simple_scatter(data, data2, r'$\chi_{AA}$', 'grid_size', None, odir, 'chi_AA_vs_grid_size.png')

            # grid_size vs chi_bb
            data = []
            for chi in chi_list:
                data.append(chi[1,1])

            # remove outliers
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            width = 1.5
            l_cutoff = np.percentile(data, 25) - width * iqr
            u_cutoff = np.percentile(data, 75) + width * iqr

            delete_arr = np.logical_or(data < l_cutoff, data > u_cutoff)
            data = np.delete(data, delete_arr, axis = None)
            data2 = np.delete(grid_size_arr, delete_arr, axis = None)
            simple_scatter(data, data2, r'$\chi_{BB}$', 'grid_size', None, odir, 'chi_BB_vs_grid_size.png')

    return L_list, S_list, D_list, chi_ij_list

def grid_dist(dataset, plot=True):
    # distribution of plaid params
    samples, experimental = get_samples(dataset)
    if not experimental:
        if plot:
            raise Exception('must be experimental')
        return
    data_dir = osp.join('/home/erschultz', dataset)
    odir = osp.join(data_dir, 'grid_size_distributions')

    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    grid_size_arr = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        dir = osp.join(data_dir, f'samples/sample{sample}/none/k0/replicate1')
        if not osp.exists(dir) or len(os.listdir(dir)) == 0:
            continue

        # get grid_size
        grid_size_arr[i] = np.loadtxt(osp.join(dir, 'grid_size.txt'))[-1]

    if plot:
        # plot plaid chi parameters
        simple_histogram(grid_size_arr, 'grid size', odir,
                            'grid_size_dist.png', dist = skewnorm)

    return grid_size_arr

def plot_params_test():
    max_ent_dir = '/home/erschultz/dataset_02_04_23/samples/sample246/PCA-normalize-E/k8/replicate1'
    diag_chis = np.loadtxt(osp.join(max_ent_dir, 'chis_diag.txt'))[-1]
    ifile = osp.join(max_ent_dir, 'resources/config.json')
    with open(ifile, 'r') as f:
        config = json.load(f)

    y = calculate_diag_chi_step(config, diag_chis)

    print(y)
    x = np.arange(0, len(y))
    plt.plot(np.log(x), y, label = 'Max Ent Edit', color = 'lightblue')
    # plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    # modify_plaid_chis('dataset_02_04_23', k = 10)
    modify_maxent_diag_chi('dataset_02_04_23', 10, False)
    # for i in range(201, 202):
        # plot_modified_max_ent(i, k = 8)
    # diagonal_dist('dataset_02_04_23', 7)
    # grid_dist('dataset_01_26_23')
    # plaid_dist('dataset_02_04_23', 10, True, False, True)
    # seq_dist('dataset_01_26_23', 4, True, False, True)
    # modify_plaid_chis('dataset_11_14_22', 8)
    # plot_params_test()
