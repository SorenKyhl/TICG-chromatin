import math
import os
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch_geometric
from modify_maxent import get_samples, plaid_dist
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import BLUE_RED_CMAP, RED_BLUE_CMAP, RED_CMAP
from pylib.utils.utils import pearson_round
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import (finalize_opt,
                                                              get_base_parser)
from sequences_to_contact_maps.scripts.clean_directories import \
    clean_directories
from sequences_to_contact_maps.scripts.load_utils import load_Y
from sequences_to_contact_maps.scripts.neural_nets.utils import (
    get_dataset, load_saved_model)
from sequences_to_contact_maps.scripts.utils import triu_to_full


def molar_contact_ratio(dataset, model_ID=None, plot=True):
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

    def get_gnn_mse(model_ID):
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

    samples, experimental = get_samples(dataset)
    samples = np.array(samples)[:100] # cap at 100
    print('samples:', samples)

    N = len(samples)
    k_means_rab = np.zeros(N)
    pca_rab = np.zeros(N)
    pca_b_rab = np.zeros(N)
    pca_var = np.zeros(N)
    L1_arr = np.zeros(N)
    y_list = []
    # L_list_exp, _ = plaid_dist('dataset_01_26_23', 4, False)
    meanDist_file = osp.join(odir, 'meanDist.npy')
    if osp.exists(meanDist_file):
        meanDist_list = np.load(meanDist_file)
        found_meanDist = True
    else:
        found_meanDist = False
        meanDist_list = []
    for i, sample in enumerate(samples):
        sample_dir = osp.join(data_dir, f'samples/sample{sample}')

        y, y_diag = load_Y(sample_dir)
        y /= np.mean(np.diagonal(y))
        y_list.append(y)

        if not found_meanDist:
            meanDist_list.append(DiagonalPreprocessing.genomic_distance_statistics(y))

        if plot:
            m = len(y)

            # L1 chi
            chi = np.load(osp.join(sample_dir, 'chis.npy'))
            L1_arr[i] = np.linalg.norm(chi, ord='nuc') # sum of |singular vales|

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
    if not experimental and model_ID is not None:
        mse_dict = get_gnn_mse(model_ID)
        mse_list = []
        for i, s in enumerate(samples):
            mse = mse_dict[s]
            mse_list.append(mse)
        np.save(osp.join(odir, 'mse.npy'), mse_list)

        # make table
        with open(osp.join(odir, 'plaid_score_table.txt'), 'w') as o:
            o.write("\\begin{center}\n")
            o.write("\\begin{tabular}{|" + "c|"*5 + "}\n")
            o.write("\\hline\n")
            o.write("\\multicolumn{5}{|c|}{" + dataset.replace('_', "\_") + "} \\\ \n")
            o.write("\\hline\n")
            o.write('Sample & K\_means R(a,b) & PCA R(a,b) & PCA \% Var & MSE \\\ \n')
            o.write("\\hline\\hline\n")
            for i, mse in enumerate(mse_list):
                vals = [k_means_rab[i], pca_rab[i], pca_var[i]]
                vals = np.round(vals, 4)
                o.write(f'{samples[i]} & {vals[0]} & {vals[1]} & {vals[2]} & {mse}\\\ \n')
            o.write("\\hline\n")
            o.write("\\end{tabular}\n")
            o.write("\\end{center}\n\n")

    # plot distributions
    if plot:
        L_list, S_list, _, _ = plaid_dist(dataset, 261, 0.01, 10, False)
        S_list = [triu_to_full(S) for S in S_list]
        # plot histograms
        for arr, label in zip([k_means_rab, pca_rab, pca_b_rab, pca_var],
                                ['kmeans_Rab', 'PCA_Rab', 'PCA_binary_Rab','PCA_var']):
            np.save(osp.join(odir, label + '.npy'), arr)
            print(label)
            if not experimental and model_ID is not None:
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
        plot_n = 12
        rows=2; cols=6
        height = 6*rows; width = 3*cols
        width_ratios = [1]*cols+[0.08]
        y_arr = np.array(y_list)
        L_arr = np.array(L_list)
        S_arr = np.array(S_list)
        k_means_rab = k_means_rab
        L1_arr = L1_arr[:plot_n]
        meanDist_arr = np.array(meanDist_list)
        samples = samples

        # plot contact maps orderd by rab
        ind = np.argsort(k_means_rab[:plot_n])
        fig, ax = plt.subplots(rows, cols+1,
                                gridspec_kw={'width_ratios':width_ratios})
        fig.set_figheight(height)
        fig.set_figwidth(width)
        vmin = 0; vmax = np.mean(y_arr)
        row = 0; col=0
        for y, val, sample in zip(y_arr[ind], k_means_rab[ind], samples[ind]):
            if col == 0:
                s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                    ax = ax[row][col], cbar_ax = ax[row][-1])
            else:
                s = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                    ax = ax[row][col], cbar = False)
            s.set_title(f'Sample {sample}\nPlaid Score = {np.round(val, 1)}', fontsize = 16)
            s.set_xticks([])
            s.set_yticks([])

            col += 1
            if col == cols:
                col = 0
                row += 1

        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'y_ordered.png'))
        plt.close()

        # plot S orderd by rab
        fig, ax = plt.subplots(rows, cols+1,
                                gridspec_kw={'width_ratios':width_ratios})
        fig.set_figheight(height)
        fig.set_figwidth(width)
        vmin = np.nanpercentile(S_arr, 1)
        vmax = np.nanpercentile(S_arr, 99)
        vmax = max(vmax, vmin * -1)
        vmin = vmax * -1
        row = 0; col=0
        for S, val, sample in zip(S_arr[ind], k_means_rab[ind], samples[ind]):
            if col == 0:
                s = sns.heatmap(S, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax[row][col], cbar_ax = ax[row][-1])
            else:
                s = sns.heatmap(S, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax[row][col], cbar = False)
            s.set_title(f'Sample {sample}\nPlaid Score = {np.round(val, 1)}', fontsize = 16)
            s.set_xticks([])
            s.set_yticks([])

            col += 1
            if col == cols:
                col = 0
                row += 1

        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'S_ordered.png'))
        plt.close()

        # plot S_dag orderd by rab
        S_dag_arr = np.array([np.sign(S) * np.log(np.abs(S)+1) for S in S_arr])
        fig, ax = plt.subplots(rows, cols+1,
                                gridspec_kw={'width_ratios':width_ratios})
        fig.set_figheight(height)
        fig.set_figwidth(width)
        vmin = np.nanpercentile(S_dag_arr, 1)
        vmax = np.nanpercentile(S_dag_arr, 99)
        vmax = max(vmax, vmin * -1)
        vmin = vmax * -1
        row = 0; col=0
        for S_dag, val, sample in zip(S_dag_arr[ind], k_means_rab[ind], samples[ind]):
            if col == 0:
                s = sns.heatmap(S_dag, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax[row][col], cbar_ax = ax[row][-1])
            else:
                s = sns.heatmap(S_dag, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                    ax = ax[row][col], cbar = False)
            s.set_title(f'Sample {sample}\nPlaid Score = {np.round(val, 1)}', fontsize = 16)
            s.set_xticks([])
            s.set_yticks([])

            col += 1
            if col == cols:
                col = 0
                row += 1

        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'S_dag_ordered.png'))
        plt.close()



        # plot L_ij ordered by rab
        fig, ax = plt.subplots(rows, cols)
        fig.set_figheight(6*2)
        fig.set_figwidth(6*3)
        row = 0; col=0
        bin_width = 1
        # arr_exp = np.array(L_list_exp).reshape(-1)
        for L, val, sample in zip(L_arr[ind], k_means_rab[ind], samples[ind]):
            arr = L.reshape(-1)
            # bins = range(math.floor(min(arr_exp)), math.ceil(max(arr_exp)) + bin_width, bin_width)
            # ax[row][col].hist(arr_exp, weights = np.ones_like(arr_exp) / len(arr_exp),
            #                             bins = bins,
            #                             alpha = 0.5, label = 'Experiment')
            bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
            ax[row][col].hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = bins,
                                        alpha = 0.5, label = 'Simulation')
            ax[row][col].set_title(f'Sample {sample}\nPlaid Score = {np.round(val, 1)}', fontsize = 16)

            col += 1
            if col == cols:
                col = 0
                row += 1

        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'L_dist_ordered.png'))
        plt.close()

        # plot all meanDist
        for meanDist, sample in zip(meanDist_arr, samples):
            plt.plot(meanDist)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'meanDist.png'))
        plt.close()

        # plot meanDist colored by plaid score
        for meanDist, val, sample in zip(meanDist_arr[ind], k_means_rab[ind], samples[ind]):
            plt.plot(meanDist, label = f'sample{sample}: {np.round(val, 1)}')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'meanDist_plaid_score.png'))
        plt.close()

        # plot meanDist colored by L1 score
        for meanDist, val, sample in zip(meanDist_arr[ind], L1_arr[ind], samples[ind]):
            plt.plot(meanDist, label = f'sample{sample}: {np.round(val, 1)}')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(data_dir, 'meanDist_L1_score.png'))
        plt.close()

    np.save(meanDist_file, meanDist_list)

    return meanDist_list

if __name__ == '__main__':
    # molar_contact_ratio('dataset_01_27_23_v5', True)
    # molar_contact_ratio('dataset_01_26_23', True)
    # molar_contact_ratio('dataset_02_06_23', 363)
    # molar_contact_ratio('dataset_02_13_23', 372)
    # molar_contact_ratio('dataset_03_03_23', 387)
    # molar_contact_ratio('dataset_04_28_23', None)
    molar_contact_ratio('dataset_09_19_23', None)
