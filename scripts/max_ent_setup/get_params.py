import argparse
import json
import math
import os
import os.path as osp
import re
import subprocess as sp
import sys
from collections import defaultdict
from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_json

sys.path.append('/home/erschultz')

from sequences_to_contact_maps.scripts.argparse_utils import (
    ArgparserConverter, finalize_opt, get_base_parser)
from sequences_to_contact_maps.scripts.clean_directories import \
    clean_directories
from sequences_to_contact_maps.scripts.neural_nets.utils import (
    get_dataset, load_saved_model)


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[0]

class GetEnergy():
    def __init__(self, m):
        self.m = m

    def get_args():
        AC = ArgparserConverter()
        parser = argparse.ArgumentParser(description='Energy parser')
        parser.add_argument('--m', type=int,
                            help='number of particles')
        parser.add_argument('--gnn_model_path', type=str,
                            help='path to GNN model')
        parser.add_argument('--sample_path', type=str,
                            help='name of samples directory')
        parser.add_argument('--bonded_path', type=str,
                            help='path to bonded folder')
        parser.add_argument('--sub_dir', type=str,
                            help='name of samples directory')
        parser.add_argument('--ofile', type=str,
                            help='save file')
        parser.add_argument('--verbose', type=AC.str2bool, default=False,
                            help='True to verbose mode')
        parser.add_argument('--use_gpu', type=AC.str2bool, default=True,
                            help='True to GPU')

        args, _ = parser.parse_known_args()
        print('Args:')
        print(args)
        return args

    def get_energy_gnn(self, model_path, sample_path, kr=False, bonded_path=None,
                        sub_dir='samples', use_gpu=True, verbose=True,
                        return_plaid_diag=False, return_model_data=False):
        '''
        Loads output from GNN model to use as energy matrix

        Inputs:
            model_path: path to model results
            sample_path: path to sample

        Outputs:
            s: np array of pairwise energies
        '''
        if verbose:
            print('\nget_energy_gnn')

        # extract sample info
        sample = osp.split(sample_path)[1]
        sample_id = sample[6:]
        sample_path_split = osp.normpath(sample_path).split(os.sep)
        sample_dataset = sample_path_split[-3]

        if verbose:
            print(sample, sample_id, sample_dataset)

        # extract model info
        model_path_split = osp.normpath(model_path).split(os.sep)
        model_id = model_path_split[-1]
        model_type = model_path_split[-2]
        if verbose:
            print(f'Model type: {model_type}')
        assert model_type == 'ContactGNNEnergy', f"Unrecognized model_type: {model_type}"

        argparse_path = osp.join(model_path, 'argparse.txt')
        with open(argparse_path, 'r') as f:
            for line in f:
                if line == '--data_folder\n':
                    break
            data_folder = f.readline().strip()
            gnn_dataset = osp.split(data_folder)[1]

        if gnn_dataset == sample_dataset:
            energy_hat_path = osp.join(model_path, f"{sample}/energy_hat.txt")
            if osp.exists(energy_hat_path):
                energy = np.loadtxt(energy_hat_path)
                return energy
        else:
            print(f'WARNING: dataset mismatch: {gnn_dataset} vs {sample_dataset}')

        # set up argparse options
        parser = get_base_parser()
        sys.argv = [sys.argv[0]] # delete args from get_params, otherwise gnn opt will try and use them
        opt = parser.parse_args(['@{}'.format(argparse_path)])
        opt.id = int(model_id)
        opt = finalize_opt(opt, parser, local = True, debug = True, bonded_path=bonded_path)
        if self.m > 0:
            opt.m = self.m # override m
        opt.data_folder = osp.join('/',*sample_path_split[:-2]) # use sample_dataset not gnn_dataset
        opt.output_mode = None # don't need output, since only predicting
        opt.root_name = f'GNN{opt.id}-{sample}' # need this to be unique
        opt.log_file = sys.stdout # change
        opt.cuda = False # force to use cpu
        opt.device = torch.device('cpu')
        opt.scratch = '/home/erschultz/scratch' # use local scratch
        opt.plaid_score_cutoff = None # no cutoff at test time


        if opt.y_preprocessing.startswith('sweep'):
            _, *opt.y_preprocessing = opt.y_preprocessing.split('_')
            if isinstance(opt.y_preprocessing, list):
                opt.y_preprocessing = '_'.join(opt.y_preprocessing)
        if kr:
            opt.kr = True
            opt.y_preprocessing = f'{opt.y_preprocessing}'
        if verbose:
            print(opt)

        # get model
        model, _, _ = load_saved_model(opt, verbose=verbose)

        # get dataset
        dataset = get_dataset(opt, verbose = verbose, samples = [sample_id],
                                sub_dir = sub_dir)
        if verbose:
            print('Dataset: ', dataset, len(dataset))
            print()

        # get prediction
        data = dataset[0]
        data.batch = torch.zeros(data.num_nodes, dtype=torch.int64)

        if return_model_data:
            clean_directories(GNN_path = opt.root)
            return model, data, dataset

        if verbose:
            print(data)

        if use_gpu:
            num_it = 0
            max_it = 15
            yhat = None
            vram = get_gpu_memory()
            while yhat is None and num_it < max_it:
                print(f'GPU vram: {vram}')
                if vram > 3500:
                    model = model.to('cuda:0')
                    data = data.to('cuda:0')
                    with torch.no_grad():
                        yhat = model(data, verbose=verbose)
                else:
                    sleep(15)
                num_it += 1
                vram = get_gpu_memory()
            if num_it == max_it:
                clean_directories(GNN_path = opt.root)
                raise Exception(f'max_it exceeded: {opt.root}')
        else:
            with torch.no_grad():
                yhat = model(data, verbose=verbose)
        if torch.isnan(yhat).any():
            clean_directories(GNN_path = opt.root)
            raise Exception(f'nan in yhat: {opt.root}')

        energy = yhat.cpu().detach().numpy().reshape((opt.m,opt.m))
        if verbose:
            if use_gpu:
                print(f'Took {num_it} iterations')
            print('energy', energy)


        if opt.output_preprocesing is not None:
            if 'log' in opt.output_preprocesing:
                energy = np.multiply(np.sign(energy), np.exp(np.abs(energy)) - 1)
                # Note that ln(L + D) doesn't simplify
                # so plaid_hat and diagonal_hat can't be compared to the ground truth

            if 'center' in opt.output_preprocesing and 'norm' in opt.output_preprocesing:
                ref = np.load(osp.join(data.path, 'S.npy'))
                ref_mean = np.mean(ref)
                ref_center = ref - ref_mean
                ref_max = np.max(np.abs(ref_center))
                energy *= ref_max
                energy += ref_mean
            elif 'norm' in opt.output_preprocesing:
                ref = np.load(osp.join(data.path, 'S.npy'))
                ref_max = np.max(np.abs(ref))
                energy *= ref_max
            elif 'center' in opt.output_preprocesing:
                ref = np.load(osp.join(data.path, 'S.npy'))
                ref_mean = np.mean(ref)
                energy += ref_mean

        if verbose:
            print('energy processed', energy)


        # if plaid_hat is not None and diagonal_hat is not None and verbose:
        #     # plot plaid contribution
        #     v_max = max(np.max(energy), -1*np.min(energy))
        #     v_min = -1 * v_max
        #     # vmin = vmax = 'center'
        #     plaid_hat = plaid_hat.cpu().detach().numpy().reshape((opt.m,opt.m))
        #     plot_matrix(plaid_hat, 'plaid_hat.png', vmin = v_min,
        #                     vmax = v_max, title = 'plaid portion', cmap = 'blue-red')
        #     print(f'Rank of plaid_hat: {np.linalg.matrix_rank(plaid_hat)}')
        #     w, v = np.linalg.eig(plaid_hat)
        #     prcnt = np.abs(w) / np.sum(np.abs(w))
        #     print(prcnt[0:4])
        #     print(np.sum(prcnt[0:4]))
        #     np.savetxt('plaid_hat.txt', plaid_hat)
        #
        #
        #     # plot diag contribution
        #     diagonal_hat = diagonal_hat.cpu().detach().numpy().reshape((opt.m,opt.m))
        #     plot_matrix(diagonal_hat, 'diagonal_hat.png', vmin = v_min,
        #                     vmax = v_max, title = 'diagonal portion', cmap = 'blue-red')
        #     np.savetxt('diagonal_hat.txt', diagonal_hat)

        # cleanup
        # opt.root is set in get_dataset
        model = model.cpu()
        del model; del data; del dataset; del yhat
        torch.cuda.empty_cache()
        clean_directories(GNN_path = opt.root)
        return energy


### Tester class ###
class Tester():
    def __init__(self):
        self.dataset = 'dataset_test'
        self.sample = 1
        self.args_file = None
        self.sample_folder = osp.join('/home/erschultz', self.dataset, f'samples/sample{self.sample}')
        self.m = 1024
        self.k = 12
        self.plot = False
        self.GetSeq = GetSeq(self, None)

    def test_energy_GNN(self):
        sample = '/home/erschultz/dataset_02_04_23/samples/sample201'
        config = load_json(osp.join(sample, 'optimize_grid_b_140_phi_0.03-GNN400/config.json'))
        getEnergy = GetEnergy(config = config)

        model = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/400'
        s = getEnergy.get_energy_gnn(model, sample)
        print(s)

    def test_suite(self):
        self.test_energy_GNN()

def main():
    args = GetEnergy.get_args()
    getenergy = GetEnergy(args.m)
    S = getenergy.get_energy_gnn(args.gnn_model_path, args.sample_path,
                                bonded_path = args.bonded_path,
                                sub_dir = args.sub_dir, use_gpu = args.use_gpu,
                                verbose = args.verbose)
    np.save(args.ofile, S)


if __name__ ==  "__main__":
    main()
    # vram = get_gpu_memory()
    # print(vram)
    # cpu_prcnt = psutil.cpu_percent()
    # print(cpu_prcnt)
    # Tester().test_suite()
