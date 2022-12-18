import json
import os
import os.path as osp
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from get_goal_experimental import get_diag_goal, get_plaid_goal
from numba import njit
from scipy.ndimage import uniform_filter

for p in ['/home/erschultz', '/home/erschultz/sequences_to_contact_maps']:
    sys.path.insert(1, p)
from sequences_to_contact_maps.utils.plotting_utils import plot_matrix
from sequences_to_contact_maps.utils.similarity_measures import SCC


class Sim:
    def __init__(self, rel_path):
        self.path = osp.join(os.getcwd(), rel_path)
        self.replicate_path = osp.split(os.getcwd())[0]
        self.metrics = {}

        config_file = osp.join(self.path, 'config.json')
        if osp.exists(config_file):
            with open(config_file) as f:
                self.config = json.load(f)
                # many of these are needed for get_diag_goal
                self.k = self.config['nspecies']
                self.m = self.config["nbeads"]
                self.v_bead = self.config['beadvol']
                self.grid_size = self.config["grid_size"]
                if 'diag_chis' in self.config.keys():
                    self.diag_bins = len(self.config['diag_chis'])
                    self.diag_start = self.config['diag_start']
                    self.diag_cutoff = self.config['diag_cutoff']

                    if self.config['dense_diagonal_on']:
                        self.dense = True
                        if 'n_small_bins' in self.config.keys():
                            self.n_small_bins = self.config['n_small_bins']
                            self.small_binsize = self.config['small_binsize']
                            self.big_binsize = self.config['big_binsize']
                        else:
                            self.dense = False
                            # compatibility with Soren
                            self.n_small_bins = int(self.config['dense_diagonal_loading'] * self.diag_bins)
                            n_big_bins = self.diag_bins - self.n_small_bins
                            m_eff = self.diag_cutoff - self.diag_start # number of beads with nonzero interaction
                            dividing_line = m_eff * self.config['dense_diagonal_cutoff']
                            self.small_binsize = int(dividing_line / (self.n_small_bins))
                            self.big_binsize = int((m_eff- dividing_line) / n_big_bins)
                    else:
                        self.n_small_bins = None
                        self.small_binsize = None
                        self.big_binsize = None

        else:
            print(f"config.json missing at {config_file}")

        self.chi = self.load_chis()

        hic_file = osp.join(self.path, 'contacts.txt')
        if osp.exists(hic_file):
            self.hic = np.loadtxt(hic_file)
            self.hic /= np.mean(self.hic.diagonal())
            self.d = get_diagonal(self.hic)
        else:
            print(f"{hic_file} does not exist")

        energy_file = osp.join(self.path, "energy.traj")
        if osp.exists(energy_file):
            self.energy = pd.read_csv(energy_file, sep='\t', names=["step", "bonded", "nonbonded", "diagonal", "y"])
        else:
            print(f"{energy_file} does not exist")

        self.seqs = None
        if self.k > 0:
            try:
                self.seqs = self.load_seqs()
                assert self.k == np.shape(self.seqs)[1], f'{self.k} != {np.shape(self.seqs)[0]}'
            except Exception as e:
                print(e)
                print("load seqs failed")

        gthic_path = osp.join(self.replicate_path, "resources/y_gt.npy")
        if osp.exists(gthic_path):
            self.gthic = np.load(gthic_path).astype(np.float64)
            self.gthic /= np.mean(self.gthic.diagonal())
        else:
            self.gthic = None
            print(f"{gthic_path} does not exist")


        plaid_observables_file = osp.join(self.path,"observables.traj")
        self.obs_full = None
        self.obs = None
        if osp.exists(plaid_observables_file):
            try:
                self.obs_full = pd.read_csv(plaid_observables_file, sep ='\t', header=None)
                self.obs = self.obs_full.mean().values[1:]
            except Exception as e:
                print(f'Could not load plaid observables: {e}')
        else:
            print(f"{plaid_observables_file} does not exist")

        diag_observables_file = osp.join(self.path, "diag_observables.traj")
        if osp.exists(diag_observables_file) and osp.getsize(diag_observables_file) > 0:
            self.diag_obs_full = pd.read_csv(diag_observables_file, sep ='\t', header=None)
            self.diag_obs = self.diag_obs_full.mean().values[1:]
        else:
            self.diag_obs_full = None
            self.diag_obs = None
            print(f"{diag_observables_file} does not exist")

        obj_goal_path = osp.join(self.replicate_path, "obj_goal.txt")
        if osp.exists(obj_goal_path):
            self.obj_goal = np.loadtxt(obj_goal_path)
        else:
            self.obj_goal = None
            if self.obs is not None: # else fail quietly - goal not expected
                print(f"{obj_goal_path} does not exist")

        obj_goal_diag_path = osp.join(self.replicate_path, "obj_goal_diag.txt")
        if osp.exists(obj_goal_diag_path):
            self.obj_goal_diag = np.loadtxt(obj_goal_diag_path)
        else:
            self.obj_goal_diag = None
            print(f"{obj_goal_diag_path} does not exist")

        # stack goals/observables
        try:
            if self.obj_goal is None and self.obj_goal_diag is None:
                self.obj_goal_tot = None
                self.obs_tot = None
            elif self.obj_goal is None:
                self.obj_goal_tot = self.obj_goal_diag
                self.obs_tot = self.diag_obs
            elif self.obj_goal_diag is None:
                self.obj_goal_tot = self.obj_goal
                self.obs_tot = self.obs
            else:
                self.obj_goal_tot = np.hstack((self.obj_goal, self.obj_goal_diag))
                self.obs_tot = np.hstack((self.obs, self.diag_obs))
        except Exception as e:
            print(f"Could not stack goals/observables: {e}")

    def load_chis(self):
        if self.k == 0:
            return None

        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        chi = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                if j >= i:
                    chi[i,j] = self.config["chi" + letters[i] + letters[j]]
                    chi[j,i] = self.config["chi" + letters[i] + letters[j]]
        return chi

    def load_seqs(self):
        seqs = []
        for file in self.config["bead_type_files"]:
            seqs.append( np.loadtxt(self.path + "/" + file) )
        seqs = np.array(seqs)
        return seqs.T

    def plot_energy(self):
        fig, axs = plt.subplots(3,1, figsize=(12,10))
        sz = np.shape(self.energy)[0]

        last20 = int(sz - sz/5)

        bondmean = np.mean(self.energy['bonded'][:last20])
        nbondmean = np.mean(self.energy['nonbonded'][:last20])
        diagmean = np.mean(self.energy['diagonal'][:last20])

        axs[0].plot(self.energy['bonded'], label='bonded')
        axs[1].plot(self.energy['nonbonded'], label='nonbonded')
        axs[2].plot(self.energy['diagonal'], label='diagonal')
        axs[0].hlines(bondmean, 0, sz, colors='k')
        axs[1].hlines(nbondmean, 0, sz, colors='k')
        axs[2].hlines(diagmean, 0, sz, colors='k')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

    def plot_obs(self, diag):
        plt.figure()
        if self.obs_full is not None:
            o = np.array(self.obs_full.T)
            plt.plot(o[1:].T);
        if diag and self.diag_obs_full is not None:
            d = np.array(self.diag_obs_full.T)
            plt.semilogy(d[1:].T)

    def plot_tri(self, ofile, vmaxp=None, title="", log=False):
        '''
        Plot contact map with lower triangle as ground truth and upper as simulation.
        '''
        if log:
            first = np.log(self.hic + 1)
            second = np.log(self.gthic + 1)
        else:
            first = self.hic
            second = self.gthic

        assert np.shape(first) == np.shape(second)

        npixels = np.shape(first)[0]
        indu = np.triu_indices(npixels)
        indl = np.tril_indices(npixels)

        composite = np.zeros((npixels, npixels))

        composite[indu] = first[indu]
        composite[indl] = second[indl]

        if vmaxp is None:
            vmaxp = np.mean(second)

        plot_matrix(composite, ofile, title, vmax = vmaxp, triu = True)

    def plot_scatter(self):
        hic1 = self.hic
        hic2 = self.gthic
        plt.figure(figsize=(12,10))
        plt.plot(np.log10(hic1).flatten(), np.log10(hic2).flatten(), 'kx', markersize=0.1)
        plt.plot(np.log10(np.linspace(0.001,1,10)), np.log10(np.linspace(0.001,1,10)), 'r--')

        scc = SCC().scc(hic1, hic2)
        pearson = get_pearson(hic1, hic2)
        rmse = get_RMSE(hic1, hic2)

        plt.title("Scatter (raw) Pearson: {%.2f}, RMSE: {%.2f}" %(pearson, rmse))

        plt.xlabel('hic')
        plt.ylabel('gthic')
        plt.axis([-4,0.5,-5, 0.5])

    def plot_diff(self):
        first = self.hic
        second = self.gthic

        c = first.mean() + 0.1*first.std()
        plt.figure(figsize=(12,10))
        diff = first - second

        cutoffmin = -c
        cutoffmax = c
        scc = SCC().scc(first, second)
        pearson = get_pearson(first, second)
        rmse = get_RMSE(first, second)

        plt.xlabel("Pearson: {%.2f}, SCC: {%.2f}, RMSE: {%.2f}" %(pearson, scc, rmse))

        plt.imshow(diff, vmin= cutoffmin, vmax = cutoffmax, cmap = 'bwr')
        plt.colorbar()

    def plot_obs_vs_goal(self, ofile):
        fig, axs = plt.subplots(2, figsize=(12,14))
        obs = self.obs_tot
        goal = self.obj_goal_tot
        axs[0].plot(obs, '--o', label="obs")
        axs[0].plot(goal, 'ko', label="goal")
        axs[0].legend()
        axs[0].set_title("Observables vs Goals", fontsize = 16)

        diff = self.obs_tot - self.obj_goal_tot
        diff_sum = np.round(np.sum(diff), 3)
        axs[1].plot(diff, '--o')
        axs[1].hlines(0, len(self.obs_tot), 0, 'k')
        axs[1].set_title(f"Difference\nSum: {diff_sum}", fontsize = 16)
        plt.savefig(ofile)
        plt.close()

    def plot_oe(self):
        oe = get_oe(self.hic)
        plt.figure(figsize=(12,10))
        plt.imshow(oe, vmin=0, vmax=2, cmap='bwr')
        plt.colorbar()

    def plot_consistency(self):
        # can only calculate consistency if the hic resolution is the same
        # as the observables resolution
        if (self.config['contact_resolution'] == 1):
            hic = self.hic
            plt.figure()
            plt.plot(self.obs_tot, 'o', label="obs")

            diag = get_diag_goal(hic, self)
            print(diag, diag.shape)
            if self.seqs is not None:
                self.verbose = False
                plaid = get_plaid_goal(hic, self, self.seqs)
                if len(self.obs_tot) > len(plaid):
                    # heuristic check for if both obs are considered - TODO
                    goal = np.hstack((plaid, diag))
                    plt.axvline(len(plaid)-0.5)
                else:
                    goal = plaid
            else:
                goal = diag

            diff = self.obs_tot - goal
            error = np.sqrt(diff@diff / (goal@goal))

            plt.plot(goal, 'x', label="goal")
            plt.title("sim.obs vs Goals(sim.hic); Error: {%.3f}"%error)
            plt.legend()
            return error

        elif sim.config['contact_resolution'] > 1:
            print("cannot plot consistency because sim.config['contact_resolution'] > 1")
            return 0

def get_RMSE(hic1, hic2):
    diff = hic1-hic2
    rmse = np.sqrt(diff.flatten()@diff.flatten())
    return rmse

def get_pearson(hic1, hic2):
    return np.corrcoef(hic1.flatten(), hic2.flatten())[0,1]

@njit
def get_oe(contact, diagonal=None):
    """Returns observed over expected matrix.
    (normalize contact map by the mean of the sub-diagonal)
    """

    if diagonal is None:
        diagonal = get_diagonal(contact)

    oe = np.zeros_like(contact)
    for i,row in enumerate(contact):
        for j, col in enumerate(contact):
            d = diagonal[abs(i-j)]
            if d == 0:
                # division by zero is undefined
                oe[i,j] = 0
            else:
                oe[i,j] = contact[i,j]/d

    return oe

@njit
def get_diagonal(contact):
    """Returns the probablity of contact as a function of genomic distance"""
    rows, cols = np.shape(contact)
    d = np.zeros(rows)
    for k in range(rows):
        d[k] = np.mean(np.diag(contact, k))
    return d

def main():
    print("analysis")
    sim = Sim("production_out")

    if sim.gthic is not None:
        sim.plot_tri("tri.png")
        sim.plot_tri("tri_log.png", log = True)
        sim.plot_tri("tri_dark.png", np.mean(sim.gthic)/2)

        sim.plot_diff()
        plt.savefig("diff.png")

        sim.plot_scatter()
        plt.savefig("scatter.png")

    sim.plot_energy()
    plt.savefig("energy.png")

    sim.plot_obs(diag=True)
    plt.savefig("obs.png")

    try:
        plt.figure()
        plt.plot(sim.config['diag_chis'], 'o')
        plt.savefig("diag_chis.png")
    except KeyError:
        pass

    if sim.obs_tot is not None:
        # obs tot is None if 0 max ent iterations
        sim.plot_obs_vs_goal("obs_vs_goal.png")

        error = sim.plot_consistency()
        plt.savefig("consistency.png")
        if error > 0.01:
            print("SIMULATION IS NOT CONSISTENT")

def test():
    dir = '/home/erschultz/dataset_11_14_22/samples/sample2201/PCA-normalize-E/k12/replicate1/iteration16'
    os.chdir(dir)
    sim = Sim("production_out")
    sim.plot_tri("tri.png")
    sim.plot_tri("tri_log.png", log = True)

if __name__ == '__main__':
    main()
    # test()
