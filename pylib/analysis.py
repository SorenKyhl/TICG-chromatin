import csv
import json
import logging
import os
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylib.utils.epilib as ep
import pylib.utils.utils as utils
from codetiming import Timer
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_all_energy
from pylib.utils.plotting_utils import (plot_matrix, plot_mean_dist,
                                        plot_mean_vs_genomic_distance)
from pylib.utils.similarity_measures import SCC
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

matplotlib.use('Agg')
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams.update({"font.size": 18})



#@Timer("sim analysis took {:.2f} seconds", logger=logging.info)
def sim_analysis(sim, fast_analysis=False):
    """analyze data from simulation only (doesn't require ground truth hic)"""
    error = sim.plot_consistency()
    if error is None:
        return
    if error > 0.01:
        logging.error("SIMULATION IS NOT CONSISTENT")

    sim.plot_contactmap()
    np.save('y.npy', sim.hic)

    plt.figure()
    sim.plot_energy()
    plt.savefig("energy.png")
    plt.close()

    plt.figure()
    sim.plot_obs(diag=False)
    plt.close()

    if sim.config['diagonal_on']:
        plt.figure()
        plt.plot(sim.config["diag_chis"], "o")
        plt.savefig("diag_chis.png")
        plt.close()

    if sim.config['nspecies'] > 0:
        plt.figure()
        utils.plot_image(sim.config["chis"])
        plt.savefig("chis.png")
        plt.close()

        plt.figure()
        plot_chi_matrix(sim)
        plt.close()

        if not fast_analysis:
            try:
                # don't run if the matrices are big
                if sim.config["nbeads"] <= 2048:
                    plot_energy_matrices(sim, save=True)
            except ValueError:
                if sim.config["contact_resolution"] > 1:
                    logging.warn("energy matrices could not be created because contact map has been pooled (contact map resolution > 1)")
                else:
                    raise ValueError
            except NotImplementedError:
                logging.warn("energy matrices not implemented for this situation")
    elif osp.exists('S.npy') and not fast_analysis:
        S = np.load('S.npy')
        plot_matrix(S, 'matrix_S.png', "S", cmap='bluered')
        meanDist_S = DiagonalPreprocessing.genomic_distance_statistics(S, mode='freq')
        plot_mean_dist(meanDist_S, '', 'meanDist_S_log.png', None,
                        logx = True, logy = False,
                        ylabel = 'mean(diagonal(S, d))')

    plt.figure()
    sim.plot_oe()
    plt.savefig("oe.png")
    plt.close()

    plt.figure()
    sim.plot_diagonal()
    plt.savefig("diagonal.png")
    plt.close()

    plt.figure()
    sim.plot_diagonal(scale="log")
    plt.savefig("diagonal-log.png")
    plt.close()

    if fast_analysis is False:
        plot_mean_vs_genomic_distance(sim.hic, '', 'meanDist_log.png',
                                        logx = True, ref = sim.gthic)


def compare_analysis(sim, fast_analysis=False):
    """analyze comparison of simulation with ground truth contact map"""

    if fast_analysis is False:
        ep.eric_plot_tri(sim.hic, sim.gthic, "tri.png")
        ep.eric_plot_tri(sim.hic, sim.gthic, "tri_log.png", log = True)
        ep.eric_plot_tri(sim.hic, sim.gthic, "tri_dark.png", np.mean(sim.gthic)/2)

        plot_dist_stratified_pearson_r(sim.hic, sim.gthic)
        compare_top_PCs(sim.hic, sim.gthic)

    plt.figure()
    ep.plot_tri(sim.hic, sim.gthic, oe=True)
    plt.savefig("tri_oe.png")
    plt.close()

    sim.plot_tri()
    plt.savefig("tri_sk.png")
    plt.close()

    sim.plot_tri(log=True)
    plt.savefig("tri_log_sk.png")
    plt.close()

    sim.plot_tri(vmaxp=np.mean(sim.hic) / 2)
    plt.savefig("tri_dark_sk.png")
    plt.close()

    sim.plot_diff()
    plt.savefig("diff.png")
    plt.close()

    sim.plot_scatter()
    plt.savefig("scatter.png")
    plt.close()


def maxent_analysis(sim):
    """analyze properties related to maxent optimization"""
    SCC = ep.get_SCC(sim.hic, sim.gthic)
    RMSE = ep.get_RMSE(sim.hic, sim.gthic)
    RMSLE = ep.get_RMSLE(sim.hic, sim.gthic)

    with open("../SCC.txt", "a") as f:
        f.write(str(SCC) + "\n")

    with open("../RMSE.txt", "a") as f:
        f.write(str(RMSE) + "\n")

    with open("../RMSLE.txt", "a") as f:
        f.write(str(RMSLE) + "\n")

    plt.figure()
    SCC_vec = np.loadtxt("../SCC.txt")
    plt.plot(SCC_vec)
    plt.xlabel("iteration")
    plt.ylabel("SCC")
    plt.savefig("../SCC.png")
    plt.close()

    plt.figure()
    RMSE_vec = np.loadtxt("../RMSE.txt")
    RMSLE_vec = np.loadtxt("../RMSLE.txt")
    plt.plot(RMSE_vec, label="RMSE")
    plt.plot(RMSLE_vec, label="RMSLE")
    plt.xlabel("iteration")
    plt.ylabel("RMSE")
    plt.savefig("../RMSE.png")
    plt.close()

    plt.figure()
    convergence = np.loadtxt("../convergence.txt")
    fig, axs = plt.subplots(3, figsize=(12, 14))
    axs[0].plot(convergence)
    axs[0].set_title("Loss")
    axs[1].plot(RMSE_vec, label="RMSE")
    axs[1].plot(RMSLE_vec, label="RMSLE")
    axs[1].set_title("RMSE/RMSLE")
    axs[1].legend()
    axs[2].plot(SCC_vec)
    axs[2].set_title("SCC")
    plt.savefig("../error.png")
    plt.close()

    plt.figure()
    sim.plot_obs_vs_goal()
    plt.savefig("obs_vs_goal.png")
    plt.close()

def plot_chi_matrix(sim):
    utils.plot_image(np.array(sim.config["chis"]))

def plot_energy_matrices(sim, save=True):
    # energy matrices
    L, D, S = calculate_all_energy(
        sim.config, sim.seqs, np.array(sim.config["chis"])
    )

    plot_matrix(L, 'matrix_L.png', "L", cmap='bluered')
    plot_matrix(D, 'matrix_D.png', "D")
    plot_matrix(S, 'matrix_S.png', "S", cmap='bluered')

    if save:
        np.save('S.npy', S)
        np.save('D.npy', D)
        np.save('L.npy', L)

def compare_top_PCs(y, yhat):
    # y
    pca_y = PCA()
    pca_y.fit(y/np.std(y))

    # yhat
    pca_yhat = PCA()
    pca_yhat.fit(yhat/np.std(yhat))

    results = [['Component Index', 'Accuracy', 'Pearson R']]

    for i in range(5):
        comp_y = pca_y.components_[i]
        sign_y = np.sign(comp_y)

        comp_yhat = pca_yhat.components_[i]
        sign_yhat = np.sign(comp_yhat)

        acc = np.sum((sign_yhat == sign_y)) / sign_y.size
        acc = max(acc, 1 - acc)
        corr, pval = pearsonr(comp_yhat, comp_y)
        corr = abs(corr)
        results.append([i, acc, corr])

    with open('PCA_results.txt', 'w', newline = '') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerows(results)

    results = np.array(results[1:]).astype(float)
    fig, ax1 = plt.subplots()
    ax1.plot(results[:, 0], results[:, 1], color = 'b')
    ax1.set_ylabel('Accuracy', color = 'b')

    ax2 = ax1.twinx()
    ax2.plot(results[:, 0], results[:, 2], color = 'r')
    ax2.set_ylabel('Pearson R', color = 'r')

    ax1.set_xlabel('Component Index')
    plt.xticks(results[:, 0])
    plt.savefig('PCA_results.png')
    plt.close()

# plotting functions
def plot_dist_stratified_pearson_r(y, yhat):
    m, _ = y.shape

    # diagonal preprocessing
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    y_diag = DiagonalPreprocessing.process(y, meanDist)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(yhat)
    yhat_diag = DiagonalPreprocessing.process(yhat, meanDist)

    triu_ind = np.triu_indices(m)
    overall_corr, _ = pearsonr(y[triu_ind], yhat[triu_ind])
    scc = SCC()
    corr_scc = scc.scc(y, yhat, var_stabilized = False)
    corr_scc_var = scc.scc(y, yhat, var_stabilized = True)
    avg_diag, corr_arr = calc_dist_strat_corr(y, yhat, mode = 'pearson',
                                            return_arr = True)

    # save correlations to json
    with open('distance_pearson.json', 'w') as f:
        temp_dict = {'overall_pearson': overall_corr,
                     'scc': corr_scc,
                     'scc_var': corr_scc_var,
                     'avg_dist_pearson': avg_diag}
        json.dump(temp_dict, f)

    # round
    corr_scc = np.round(corr_scc, 3)
    corr_scc_var = np.round(corr_scc_var, 3)
    avg_diag = np.round(avg_diag, 3)
    overall_corr = np.round(overall_corr, 3)

    # format title
    title = f'Overall Pearson Corr: {overall_corr}'
    title += f'\nMean Diagonal Pearson Corr: {avg_diag}'
    title += f'\nSCC: {corr_scc_var}'

    for log in [True, False]:
        plt.figure()
        plt.plot(np.arange(m-2), corr_arr, color = 'black')
        plt.ylim(-0.5, 1)
        plt.xlabel('Distance', fontsize = 16)
        plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
        plt.title(title, fontsize = 16)

        plt.tight_layout()
        if log:
            plt.xscale('log')
            plt.savefig('distance_pearson_log.png')
        else:
            plt.savefig('distance_pearson.png')
        plt.close()

    return overall_corr, corr_scc, corr_scc_var, avg_diag

def calc_dist_strat_corr(y, yhat, mode = 'pearson', return_arr = False):
    """
    Helper function to calculate correlation stratified by distance.

    Inputs:
        y: target
        yhat: prediction
        mode: pearson or spearman (str)

    Outpus:
        avg: average distance stratified correlation
        corr_arr: array of distance stratified correlations
    """
    if mode.lower() == 'pearson':
        stat = pearsonr
    elif mode.lower() == 'nan_pearson':
        stat = nan_pearsonr
    elif mode.lower() == 'spearman':
        stat = spearmanr

    assert len(y.shape) == 2
    n, _ = y.shape
    triu_ind = np.triu_indices(n)

    corr_arr = np.zeros(n-2)
    corr_arr[0] = np.NaN
    for d in range(1, n-2):
        # n-1, n, and 0 are NaN always, so skip
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, _ = stat(y_diag, yhat_diag)
        corr_arr[d] = corr

    avg = np.nanmean(corr_arr)
    if return_arr:
        return avg, corr_arr
    else:
        return avg

def main_no_compare(dir=None):
    if dir is not None:
        os.chdir(dir)
    assert osp.exists("production_out"), f'{os.getcwd()}'
    sim = ep.Sim("production_out")
    logging.info("sim created")
    sim_analysis(sim)
    logging.info("sim analysis done")

def main_no_maxent(dir=None):
    if dir is not None:
        os.chdir(dir)
    assert osp.exists("production_out"), f'{os.getcwd()}'
    sim = ep.Sim("production_out")
    logging.info("sim created")
    sim_analysis(sim)
    logging.info("sim analysis done")
    compare_analysis(sim)
    logging.info("compare analysis done")

def main(fast_analysis = False, dir = None, mode = 'all'):
    if dir is not None:
        os.chdir(dir)
    assert osp.exists("production_out"), f'{dir}, {os.getcwd()}'
    sim = ep.Sim("production_out", mode=mode)
    logging.info("sim created")
    sim_analysis(sim, fast_analysis)
    logging.info("sim analysis done")
    compare_analysis(sim, fast_analysis)
    logging.info("compare analysis done")
    maxent_analysis(sim)
    logging.info("maxent analysis done")
    plt.close('all')

if __name__ == "__main__":
    main()
