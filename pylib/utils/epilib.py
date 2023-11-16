import copy
import json
import logging
import math
import os
import sys
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from numba import njit
from pylib.utils import hic_utils, utils
from pylib.utils.goals import *
from pylib.utils.hic_utils import get_diagonal
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.similarity_measures import *
from sklearn.decomposition import PCA, KernelPCA
from tqdm import tqdm

# import palettable
# from palettable.colorbrewer.sequential import Reds_3


mycmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "custom", [(0, "white"), (0.3, "white"), (1, "#ff0000")], N=126
)


class Sim:
    """simulation analysis

    Attributes:
        metrics: cache for expensive metrics
    """

    def __init__(self, path, maxent_analysis=True, mode='all'):
        """
        Args:
            path: directory containing simulation output
            maxent_analysis: if false, won't try to load things related to maxent
        """
        self.path = Path(path)
        self.metrics = {}
        self.maxent_analysis = maxent_analysis

        try:
            self.config = self.load_config()
        except FileNotFoundError:
            logging.error("error loading config.json")

        found = False
        if osp.exists(self.path / "contacts.txt"):
            self.hic = get_contactmap(self.path / "contacts.txt")
            found = True
        elif osp.exists(self.path / f"contacts{self.config['nSweeps']}.txt"):
            self.hic = get_contactmap(self.path / f"contacts{self.config['nSweeps']}.txt")
            found = True

        if found:
            self.d = hic_utils.get_diagonal(self.hic)
        else:
            logging.error("error loading contactmap.")
            logging.error(self.path / f"contacts{self.config['nSweeps']}.txt")

        try:
            self.energy = pd.read_csv(
                self.path / "energy.traj",
                sep="\t",
                header=0,
                names=["step", "bonded", "nonbonded", "diagonal", "boundary", "total"],
                index_col=0
            )
            # for production runs, when energy trajectories are concatenated
            self.energy = self.energy.reset_index()
        except FileNotFoundError:
            logging.error("error loading energy.traj")

        self.chi = None
        self.seqs = None
        self.obs_full = None
        self.obs_tot = None
        self.obs = None
        if self.config['plaid_on']:
            self.k = self.config['nspecies']
            if self.k > 0:
                self.chi = self.load_chis()
                try:
                    self.seqs = self.load_seqs()
                    assert self.k == np.shape(self.seqs)[0]
                except OSError:
                    logging.error("error loading sequences")

                if mode != 'diag':
                    observables_file = self.path / "observables.traj"
                    if os.stat(observables_file).st_size > 0:
                        try:
                            self.obs_full = pd.read_csv(observables_file, sep="\t",
                                                        header=None)
                            self.obs = np.array(self.obs_full.mean().values[1:])
                        except FileNotFoundError:
                            logging.error("error loading plaid observables")
        else:
            self.k = 0

        self.diag_obs = None
        if self.config['diagonal_on']:
            self.diag_bins = np.shape(self.config["diag_chis"])[0]
            diag_observables_file = self.path / "diag_observables.traj"
            if os.stat(diag_observables_file).st_size > 0:
                try:
                    self.diag_obs_full = pd.read_csv(diag_observables_file, sep="\t",
                                                    header=None)
                    self.diag_obs = np.array(self.diag_obs_full.mean().values[1:])
                except FileNotFoundError:
                    logging.error(f"error loading diag observables: not found at {diag_observables_file}")

        try:
            if self.obs is None:
                self.obs_tot = self.diag_obs
            else:
                self.obs_tot = np.hstack((self.obs, self.diag_obs))
        except AttributeError:
            logging.error(f"observables not loaded: obs is {type(self.obs)}, diag_obs is {type(self.diag_obs)}, obs_tot is {type(self.obs_tot)}")

        try:
            self.extra = np.loadtxt(self.path / "extra.traj")
            self.beadvol = self.config["beadvol"]
            self.nbeads = self.config["nbeads"]
        except:
            logging.error("error loading extra observables")

        resources_path = self.path / "../../resources/"
        if resources_path.exists():
            self.resources_path = resources_path

        self.gthic = None
        if self.maxent_analysis:
            """look for maxent related things"""
            gthic_possibilites = [".", "..", "../../resources"]
            gthic_loaded = False
            for gtp in gthic_possibilites:
                gthic_path = self.path / gtp / "experimental_hic.npy"
                if os.path.exists(gthic_path) and not gthic_loaded:
                    self.gthic = np.load(gthic_path)
                    gthic_loaded = True

            if not gthic_loaded:
                logging.error("no ground truth hic found")

            # if contact map pools, also pool sequences
            if gthic_loaded and self.config["contact_resolution"] > 1:
                self.seqs = hic_utils.pool_seqs(self.seqs, self.config["contact_resolution"])
                self.gthic = hic_utils.pool_sum(self.gthic, self.config["contact_resolution"])

            obj_goal_path = resources_path / "obj_goal.txt"
            if obj_goal_path.exists():
                self.obj_goal = np.loadtxt(obj_goal_path)
            else:
                logging.error("no path to obj_goal.txt")
                logging.error(f"looking in obj_goal_path: {obj_goal_path}")

            obj_goal_diag_path = resources_path / "obj_goal_diag.txt"
            if obj_goal_diag_path.exists():
                self.obj_goal_diag = np.loadtxt(obj_goal_diag_path)
            else:
                logging.error("no path to obj_goal_diag.txt")
                logging.error(f"looking in obj_goal_diag_path: {obj_goal_path}")

            try:
                self.obj_goal_tot = np.hstack((self.obj_goal, self.obj_goal_diag))
            except AttributeError:
                try:
                    params_path = resources_path / "params.json"
                    params = utils.load_json(params_path)
                    self.obj_goal_tot = params["goals"]
                except FileNotFoundError:
                    logging.error("no maximum entropy goals found")

    def pearson(self):
        if "pearson" in self.metrics:
            return self.metrics["pearson"]
        else:
            return get_pearson(self.hic, self.gthic)

    def symmetry_score(self):
        if "symmetry" in self.metrics:
            return self.metrics["symmetry"]
        else:
            return get_pearson(self.hic, self.gthic)

    def SCC(self):
        if "SCC" in self.metrics:
            return self.metrics["SCC"]
        else:
            return get_SCC(self.hic, self.gthic)

    def RMSE(self):
        if "RMSE" in self.metrics:
            return self.metrics["RMSE"]
        else:
            return get_RMSE(self.hic, self.gthic)

    def RMSLE(self):
        if "RMSLE" in self.metrics:
            return self.metrics["RMSLE"]
        else:
            return get_RMSLE(self.hic, self.gthic)

    def load_config(self):
        with open(self.path / "config.json") as f:
            config = json.load(f)
        return config

    def load_chis(self):
        try:
            # old version of chis, each stored with their own key
            nspecies = self.config["nspecies"]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            chi = np.zeros((nspecies, nspecies))
            for i in range(nspecies):
                for j in range(nspecies):
                    if j >= i:
                        chi[i, j] = self.config["chi" + letters[i] + letters[j]]
                        chi[j, i] = self.config["chi" + letters[i] + letters[j]]
        except KeyError:
            # new version of chis, stored in a matrix
            indices = np.triu_indices(self.config["nspecies"])
            chi = np.array(self.config["chis"])[indices]
        return chi

    def save_chis(self):
        """old format of storing chis, deprecated"""
        nspecies = self.config["nspecies"]
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for i in range(nspecies):
            for j in range(nspecies):
                if j >= i:
                    self.config["chi" + letters[i] + letters[j]] = self.chi[i, j]
                    self.config["chi" + letters[i] + letters[j]] = self.chi[j, i]

    def load_seqs(self):
        seqs = []
        for file in self.config["bead_type_files"]:
            seqs.append(np.loadtxt(self.path / file))
        seqs = np.array(seqs)
        return seqs

    def get_smatrix(self):
        """
        Non-diagonal entries in chi need to be multiplied by 0.5
        to account for factor of 2 in cross terms
        """

        chimatrix = copy.deepcopy(self.chi)
        d = copy.deepcopy(chimatrix.diagonal())
        chimatrix *= 0.5
        np.fill_diagonal(chimatrix, d)
        smatrix = self.seqs.T @ chimatrix @ self.seqs

        return smatrix

    def plot_contactmap(self, vmaxp=0.1, absolute=False, dark=False,
                        title="", log=False):
        plot_contactmap(self.hic, vmaxp, absolute, dark=dark, title=title, log=log,
                        ofile="contactmap.png")

    def plot_energy(self):
        plot_energy(self)

    def plot_obs(self, diag):
        if self.obs_full is not None:
            plot_obs(self, diag, 'obs.png')

    def plot_tri(self, vmaxp=None, title="", dark=False, log=False):
        plot_tri(self.hic, self.gthic, vmaxp, title=title, dark=dark, log=log)

    def plot_scatter(self):
        plot_scatter(self.hic, self.gthic)

    def plot_diff(self):
        plot_diff(self.hic, self.gthic)

    def plot_obs_vs_goal(self):
        plot_obs_vs_goal(self)

    def plot_oe(self, log=False):
        plot_oe(get_oe(self.hic), log=log)

    def plot_consistency(self):
        if self.seqs is None:
            return 0
        return plot_consistency(self, "consistency.png")

    def calc_goals(self):
        plaid = get_goal_plaid(self.gthic, self.seqs, self.config)
        diag = get_goal_diag(
            self.gthic,
            self.config,
            adj=True,
            dense_diagonal_on=self.config["dense_diagonal_on"],
        )

        np.savetxt("obj_goal.txt", plaid, newline=" ", fmt="%.8f")
        np.savetxt("obj_goal_diag.txt", diag, newline=" ", fmt="%.8f")

    def plot_diagonal(self, *args, scale="semilogy"):
        if scale == "semilogy":
            plot_fn = plt.semilogy
        elif scale == "log" or scale == "loglog":
            plot_fn = plt.loglog
        else:
            raise ValueError("usage: scale: ['semilogy' | 'log']")

        diag = self.d
        plot_fn(np.linspace(1 / len(diag), 1, len(diag)), diag, *args, label="sim")
        if self.gthic is not None:
            diag = hic_utils.get_diagonal(self.gthic)
            plot_fn(np.linspace(1 / len(diag), 1, len(diag)), diag, "k", label="exp")

        plt.xlabel("genomic distance")
        plt.ylabel("probability")
        plt.legend()
        plt.title("probability vs genomic distance")
        lower_bound = np.max([np.min(diag) / 5, 1e-7])
        plt.axis([0, 1, lower_bound, 1])



def get_contactmap(filename, norm=True, log=False, rawcounts=False, normtype="mean"):
    """Load contact map from file"""

    contactmap = np.loadtxt(filename)

    if rawcounts:
        return contactmap
    else:
        if norm:
            if normtype == "max":
                contactmap /= np.max(np.diagonal(contactmap))
            elif normtype == "mean":
                contactmap /= np.mean(np.diagonal(contactmap))
                # np.fill_diagonal(contactmap, 1)
            # df /= df[0][0]
        if log:
            contactmap = np.log(contactmap)

        return contactmap



@njit
def get_oe(contact, diagonal=None):
    """calculate observed over expected matrix.
    (normalize contact map by the mean of the sub-diagonal)
    """

    if diagonal is None:
        diagonal = hic_utils.get_diagonal(contact)

    oe = np.zeros_like(contact)
    for i, row in enumerate(contact):
        for j, col in enumerate(contact):
            d = diagonal[abs(i - j)]
            if d == 0:
                # division by zero is undefined
                oe[i, j] = 1
            else:
                oe[i, j] = contact[i, j] / d

    return oe


def plot_oe(oe, title="", log=False):
    plt.figure(figsize=(12, 10))
    plt.imshow(oe, vmin=0, vmax=2, cmap="bwr")
    plt.title(title)
    plt.colorbar()


def plot_contactmap(contact, vmaxp=0.1, absolute=False, imshow=True, cbar=True,
                    dark=False, title="", log=False, ofile=None):
    plt.figure(figsize=(12, 10))

    cmap = mycmap
    vmin = 0

    if dark:
        vmaxp = np.mean(contact) / 2
        absolute = True

    if absolute:
        vmax = vmaxp
    else:
        vmax = contact.mean() + vmaxp * contact.std()

    if imshow:
        plot_fn = plt.imshow
    else:
        plot_fn = sns.heatmap

    if log:
        contact = np.log10(contact + 1e-20)
        vmin = np.min(contact[np.where(contact > -19)])
        vmax = vmin / 3
        # print("vmin", vmin)
        # print("vmax", vmax)

    plot_fn(contact, cmap=cmap, vmin=vmin, vmax=vmax)

    if cbar:
        plt.colorbar()
        plt.title(title)

    if ofile is not None:
        plt.savefig(ofile)
        plt.close()


def plot_diagonal(input, *args, scale="semilogy", label=None):
    if input.ndim == 2:
        # input is a full matrix
        diag = hic_utils.get_diagonal(input)
    else:
        # input is p(s)
        diag = input

    if scale == "semilogy":
        plt.semilogy(np.linspace(1 / len(diag), 1, len(diag)), diag, *args, label=label)
    elif scale == "loglog":
        plt.loglog(np.linspace(1 / len(diag), 1, len(diag)), diag, *args, label=label)


def plot_energy(sim):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    sz = np.shape(sim.energy)[0]
    last20 = int(sz - sz / 5)

    try:
        bondmean = np.mean(sim.energy["bonded"][:last20])
        nbondmean = np.mean(sim.energy["nonbonded"][:last20])
        diagmean = np.mean(sim.energy["diagonal"][:last20])
    except TypeError:
        return

    rightedge = len(sim.energy["bonded"])

    axs[0].plot(sim.energy["bonded"], label="bonded")
    axs[1].plot(sim.energy["nonbonded"], label="nonbonded")
    axs[2].plot(sim.energy["diagonal"], label="diagonal")
    axs[0].hlines(bondmean, 0, rightedge, colors="k")
    axs[1].hlines(nbondmean, 0, rightedge, colors="k")
    axs[2].hlines(diagmean, 0, rightedge, colors="k")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()


def plot_obs(sim, diag=True, ofile=None):
    o = np.array(sim.obs_full.T)
    d = np.array(sim.diag_obs_full.T)
    plt.plot(o[1:].T)
    if diag:
        plt.figure()
        plt.semilogy(d[1:].T)

    if ofile is not None:
        plt.savefig(ofile)
        plt.close()


def plot_diff(first, second, c=None, log=False, oe=False, title=""):
    if c is None:
        if log is False:
            c = first.mean() + 0.1 * first.std()
        else:
            raise NotImplementedError

    if log:
        print("log is true")
        frac = np.log(first + 1e-20), np.log(second + 1e-20)
        plt.imshow(frac)
        return

    plt.figure(figsize=(12, 10))
    diff = first - second

    if oe:
        vmin = 0.5
        vmax = 1.5
    else:
        vmin = -c
        vmax = c

    scc = get_SCC(first, second)
    pearson = get_pearson(first, second)
    rmse = get_RMSE(first, second)

    plt.title(title)
    plt.xlabel("Pearson: {%.4f}, SCC: {%.4f}, RMSE: {%.4f}" % (pearson, scc, rmse))

    plt.imshow(diff, vmin=vmin, vmax=vmax, cmap="bwr")
    plt.colorbar()

    return scc


def randomized_svd(X, r, q=10, p=1):
    """randomized svd for decomposing large matrices
    X: ndarray to decompose
    r: number of singular values
    q: oversampling
    p: power iterations"""
    ny = X.shape[1]
    P = np.random.randn(ny, r + p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z, mode="reduced")

    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY

    return U, S, VT

def get_pcs(input, k, normalize=False, binarize=False, scale=False,
            use_kernel=False, kernel=None, manual=False, soren=False,
            randomized=False, smooth=False, h=3, align=False):
    '''
    Defines seq based on PCs of input.

    Inputs:
        input: matrix to perform PCA on
        normalize: True to normalize pcs to [-1, 1]
        binarize: True to binarize pcs (will normalize and then set any value > 0 as 1)
        use_kernel: True to use kernel PCA
        kernel: type of kernel to use
        align: flip sign such that all pcs start positive

    Outputs:
        seq: array of particle types
    '''
    m = len(input)
    if binarize:
        normalize = True # reusing normalize code in binarize
    if smooth:
        input = scipy.ndimage.gaussian_filter(input, (h, h))

    if use_kernel:
        assert kernel is not None
        pca = KernelPCA(kernel = kernel)
    else:
        pca = PCA()


    if manual:
        C = np.corrcoef(input)
        W, V = np.linalg.eig(C)
        V = V.T
    elif soren:
        U, S, V = scipy.linalg.svd(np.corrcoef(input))
    else:
        if scale:
            pca.fit(input/np.std(input, axis = 0))
        else:
            pca.fit(input)
        if use_kernel:
            V = pca.eigenvectors_.T
        else:
            V = pca.components_

    seq = np.zeros((m, k))
    for j in range(k):
        pc = V[j]
        if normalize:
            val = np.max(np.abs(pc))
            # multiply by scale such that val x scale = 1
            scale = 1/val
            pc *= scale

        if binarize:
            # pc has already been normalized to [-1, 1]
            pc[pc < 0] = 0
            pc[pc > 0] = 1

        seq[:,j] = pc

    if align:
        cutoff = int(0.2 * m)
        for j in range(k):
            seq[:, j] *= np.sign(np.mean(seq[:cutoff, j]))

    return seq


def get_sequences(
    hic,
    k,
    dtype="float",
    map01=False,
    map11=True,
    split=False,
    randomized=False,
    scaleby_singular_values=False,
    scaleby_sqrt_singular_values=False,
    print_singular_values = True,
):
    """
    calculate polymer bead sequences using k principal components
    returns 2*k epigenetic sequencess

    scaleby_singular_values: (bool): scales principal components by their singular value
    """
    OEmap = get_oe(hic)
    if randomized:
        print("getting sequences with RANDOMIZED SVD")
        U, S, VT = randomized_svd(np.corrcoef(OEmap), 2 * k)
    else:
        print("getting sequences with np.linalg.svd")
        U, S, VT = np.linalg.svd(np.corrcoef(OEmap), full_matrices=0)

    # return VT can return here if you want

    pcs = []
    for i in range(k):
        if split:
            pcs.append(np.array([positive(n) for n in VT[i]]))
            pcs.append(-np.array([negative(n) for n in VT[i]]))
        else:
            pcs.append(np.array(VT[i]))
            if map11:
                for i, pc in enumerate(pcs):
                    pcs[i] = pc / max(abs(pc))

    if scaleby_singular_values:
        assert split is False
        for i, pc in enumerate(pcs):
            pc *= S[i]

    if scaleby_sqrt_singular_values:
        assert split is False
        for i, pc in enumerate(pcs):
            pc *= np.sqrt(S[i])

    assert dtype == "int" or dtype == "float"


    if print_singular_values:
        logging.info("singular values are:")
        logging.info(S)

    beads_per_bin = 1
    seqs = []
    for pc in pcs:
        seqs.append(seq_from_pc(pc, beads_per_bin, dtype, map01))

    return np.array(seqs)


def map_0_1(signal, backfrom=0):
    """Returns signal data mapped to the interval [0,1]"""
    signal = signal.real
    # signal = (signal - np.min(signal[:-backfrom]))
    # /(np.max(signal) - np.min(signal[:-backfrom]))
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # signal[signal<0] = 0
    return signal


def map_0_1_chip(signal, backfrom=0):
    """Returns signal data mapped to the interval [0,1]"""
    signal = signal.real
    signal = (signal - np.min(signal[:-backfrom])) / (
        np.max(signal) - np.min(signal[:-backfrom])
    )
    signal[signal < 0] = 0
    signal *= 1.5
    signal[signal > 1] = 1
    return signal


def get_baseline_signal(signal):
    bin_populations, bin_signal = np.histogram(signal, bins=100)
    baseline_signal = bin_signal[np.argmax(bin_populations)]
    return baseline_signal


def new_map_0_1_chip(signal):
    baseline = get_baseline_signal(signal)
    signal = (signal - baseline) / (np.max(signal) - baseline)
    signal[signal < 0] = 0
    signal *= 1.5
    signal[signal > 1] = 1
    return signal


def check_baseline(chipseq):
    """plots chipseq and the calculalted baseline signal"""
    baseline_signal = get_baseline_signal(chipseq)
    plt.plot(chipseq)
    plt.hlines(baseline_signal, 0, 1024, "r")
    print(baseline_signal)


def seq_from_pc(pc, beads_per_bin=1, dtype="int", map01=True):
    np.random.seed(1)

    if map01:
        seq = map_0_1(pc)
    else:
        seq = pc

    sequence = []
    for element in seq:
        for i in range(beads_per_bin):
            if dtype == "int":
                if np.random.rand() < element:
                    sequence.append(1)
                else:
                    sequence.append(0)
            elif dtype == "float":
                sequence.append(element)
    return sequence


def positive(f):
    """returns f(x) < 0 ? 0"""
    if f < 0:
        return 0
    else:
        return f


def negative(f):
    """returns f(x) > 0 ? 0"""
    if f > 0:
        return 0
    else:
        return f


def plot_smooth(data, n_steps=20, llabel="none", fmt="o", norm=True):
    """Returns data smoothed in a window of size n_steps."""
    time_series_df = pd.DataFrame(data)
    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    # Plotting:
    if norm:
        s = np.array(range(len(data))) / len(data)
    else:
        s = np.array(range(len(data)))
    plt.plot(s, smooth_path, fmt, linewidth=2, label=llabel)  # mean curve.
    plt.fill_between(s, under_line, over_line, color="b", alpha=0.1)  # pyright: ignore


def process(folder, ignore=100):
    data_out_filename = "data_out"
    contact = get_contactmap("./" + folder + "/" + data_out_filename + "/contacts.txt")
    # seq1 = np.array(pd.read_csv(folder + "/seq1.txt"))
    # seq2 = np.array(pd.read_csv(folder + "/seq2.txt"))
    seq1 = np.array(pd.read_csv(folder + "/pc1.txt"))
    seq2 = np.array(pd.read_csv(folder + "/pc2.txt"))

    goal_exp = np.zeros((2, 2))
    goal_exp[0, 0] = np.mean((np.outer(seq1, seq1) * contact).flatten())
    goal_exp[1, 0] = np.mean((np.outer(seq1, seq2) * contact).flatten())
    goal_exp[1, 1] = np.mean((np.outer(seq2, seq2) * contact).flatten())

    obs = pd.read_csv(
        "./" + folder + "/" + data_out_filename + "/observables.traj",
        sep="\t",
        header=None,
    )
    goal_sim = np.zeros((2, 2))

    goal_sim[0, 0] = np.mean(obs[1][ignore:])
    goal_sim[1, 0] = np.mean(obs[2][ignore:])
    goal_sim[1, 1] = np.mean(obs[3][ignore:])

    print("goals from experiment")
    print("calculated as weighted average from contact map")
    print(goal_exp)
    print("goals from simulation")
    print("calculated as mean of observables")
    print(goal_sim)


def process2(folder, ignore=100):
    contact = get_contactmap("./" + folder + "/contacts.txt")

    seqs = []
    filenames = os.popen("ls " + folder + "/pc*").read().split("\n")[:-1]
    print(filenames)
    for file in filenames:
        seqs.append(np.loadtxt(file))

    k = len(seqs)
    goal_exp = np.zeros((k, k))
    for i, seqi in enumerate(seqs):
        for j, seqj in enumerate(seqs):
            goal_exp[i, j] = np.mean((np.outer(seqi, seqj) * contact).flatten())

    obs = pd.read_csv("./" + folder + "/observables.traj", sep="\t", header=None)

    # remove first column corresponding to MC iteration
    goal_sim = np.zeros((k, k))

    # for i in range(k):
    # for j in range(i, k):
    # goal_sim[i,j] = np.mean(obs[1 + i*k + j][ignore:])

    goal_sim[0, 0] = np.mean(obs[1])
    goal_sim[1, 0] = np.mean(obs[2])
    goal_sim[0, 1] = np.mean(obs[2])
    goal_sim[1, 1] = np.mean(obs[3])

    print("goals from experiment")
    print("calculated as weighted average from contact map")
    print(goal_exp)
    print("goals from simulation")
    print("calculated as mean of observables")
    print(goal_sim)


def mask_vs_diagonal(contact, nbeads=1024, bins=16):
    raise NotImplementedError("deprecated")
    diag_exp = hic_utils.get_diagonal(contact)
    diag_exp = downsample(diag_exp, int(nbeads / bins))

    diag_mask, correction = mask_diagonal(contact, bins)

    plt.plot(np.log10(diag_exp), "--o")
    plt.plot(np.log10(diag_mask / correction), "--o")


def compare_diagonal(
    filename, nbeads=1024, bins=16, plot=True, getgoal=False, fudge_factor=1
):
    raise NotImplementedError("deprecated")
    print("diag_sim is volume fraction, average of diag_observables.traj")
    print(
        "diag_exp is p(s), average of each subdiagonal which is then",
        "downsapled to match resolution of diagonal bins",
    )
    print("diag_mask is mask, weighted average of contact map")
    print("corrected is mask, with correction factor to match p(s)")
    df = pd.read_csv(filename + "/diag_observables.traj", sep="\t", header=None)
    diag_sim = df.mean()[1:]
    diag_sim = np.array(diag_sim)

    if getgoal:
        df_goal = pd.read_csv(
            filename + "../../obj_goal_diag.txt", header=None, sep=" "
        )
        diag_sim_goal = df_goal.mean()[1:]
        diag_sim_goal = np.array(diag_sim_goal)

    diag_std = df.std()[1:]
    diag_std = np.array(diag_std)
    # plt.plot(diag_sim, 'o')

    contact = get_contactmap(filename + "/contacts.txt")
    diag_exp = hic_utils.get_diagonal(contact)
    diag_exp = downsample(diag_exp, int(nbeads / bins))

    diag_mask, correction = mask_diagonal(contact, bins)

    if plot:
        plt.figure(figsize=(12, 10))
        plt.plot(np.log10(diag_sim), "--o", label="volume_fraction")
        plt.plot(np.log10(diag_exp), "--o", label="p(s)")
        plt.plot(np.log10(diag_mask * fudge_factor), "--o", label="mask")
        # plt.plot(np.log10(diag_mask), '--o', label="mask")
        plt.plot(np.log10(diag_mask / correction), "--o", label="corrected")

        plt.plot(
            np.log10(diag_sim / fudge_factor / correction),
            "--o",
            label="volfrac converted",
        )
        if getgoal:
            plt.plot(np.log10(diag_sim_goal), label="vol frac goal")
        plt.legend()
        plt.xlabel("s")
        plt.ylabel("probability")

    # print(diag_sim_goal)
    # print(diag_sim/diag_mask)
    print(np.mean(diag_sim / diag_mask))

    return diag_sim, diag_exp, diag_mask, correction

def downsample(sequence, res):
    """
    take sequence of numbers and reduce length
    res: new step size
    """
    # assert(len(sequence)%res == 0)
    new = []
    for i in range(0, len(sequence), res):
        new.append(np.mean(sequence[i : i + res]))

    return np.array(new)

def bin_chipseq(df, resolution, method="max"):
    step = resolution
    areas = []
    for i in range(int(min(df["start"])), int(max(df["start"])), step):
        sliced = df[(df["start"] > i) & (df["start"] < i + step)]
        if method == "area":
            areas.append(np.trapz(sliced["value"], x=sliced["start"]))
        if method == "max":
            areas.append(sliced["value"].max())
        if method == "mean":
            areas.append(sliced["value"].mean())
    return areas


def plot_scatter(hic1, hic2, label1="first", label2="second"):
    plt.figure(figsize=(12, 10))
    plt.plot(np.log10(hic1).flatten(), np.log10(hic2).flatten(), "kx", markersize=0.1)
    plt.plot(
        np.log10(np.linspace(0.001, 1, 10)), np.log10(np.linspace(0.001, 1, 10)), "r--"
    )

    scc = get_SCC(hic1, hic2)
    pearson = get_pearson(hic1, hic2)
    rmse = get_RMSE(hic1, hic2)

    plt.title("Scatter (raw) Pearson: {%.2f}, RMSE: {%.2f}" % (pearson, rmse))

    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.axis([-4, 0.5, -5, 0.5])


def plot_scatter_oe(hic1, hic2):
    scc = get_SCC(hic1, hic2)

    hic1 = get_oe(hic1)
    hic2 = get_oe(hic2)

    plt.figure(figsize=(12, 10))
    plt.plot(np.log10(hic1).flatten(), np.log10(hic2).flatten(), "kx", markersize=0.1)
    plt.plot(
        np.log10(np.linspace(0.001, 100, 10)),
        np.log10(np.linspace(0.001, 100, 10)),
        "r--",
    )

    plt.title("Scatter (normalized) SCC: {%.2f}" % (scc))
    plt.xlabel("log(OE_sim)")
    plt.ylabel("log(OE_exp)")
    plt.axis([-4, 0.5, -5, 0.5])
    plt.axis([-2, 2, -2, 2])


def fill_subdiagonal(a, offset, fn):
    n = np.shape(a)[0]
    inds = np.arange(n - offset)
    orig_subdiag = np.diag(a, offset)
    new_subdiag = fn(orig_subdiag)
    a[inds, inds + offset] = new_subdiag
    a[inds + offset, inds] = new_subdiag
    return a


def resize_contactmap(hic, sizex, sizey):
    raise Exception("Not Implemented")

def plot_consistency(sim, ofile=None):
    """ensure simulation observables are consistent with goals
    computed from simulation contact map"""
    if len(sim.hic) != len(sim.seqs[0]):
        size = np.shape(sim.seqs[0])[0]
        hic = resize_contactmap(sim.hic, size, size)
    else:
        hic = sim.hic

    goal = get_goals(hic, sim.seqs, sim.config)

    if sim.obs_tot is None or goal is None:
        print('sim.obs_tot is None or goal is None')
        return 0

    if len(sim.obs_tot)  < len(goal):
        N = len(sim.obs_tot)
        goal = goal[-N:]
    diff = sim.obs_tot - goal
    error = np.sqrt(diff @ diff / (goal @ goal))
    if error > 0.01:
        ratio = sim.obs_tot / goal
        print(f'ratio: {ratio}')

    plt.figure()
    plt.plot(sim.obs_tot, "o", label="obs")
    plt.plot(goal, "x", label="goal")
    plt.title("sim.obs vs Goals(sim.hic); Error: {%.3f}" % error)
    plt.legend()
    if ofile is not None:
        plt.savefig(ofile)
        plt.close()
    return error

def get_symmetry_score_2(first, second, order="fro"):
    composite = make_tri_composite(first, second)
    return get_symmetry_score(composite, order)

def make_tri_composite(first, second):
    """make composite matrix with upper triangle taken from first
    and lower triangle taken from second"""
    npixels = np.shape(first)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)

    composite = np.zeros((npixels, npixels))

    composite[indu] = first[indu]
    composite[indl] = second[indl]
    return composite

def eric_plot_tri(sim, exp, ofile, vmaxp=None, title="", log=False, cmap=None):
    '''
    Plot contact map with lower triangle as ground truth and upper as simulation.
    '''
    assert np.shape(sim) == np.shape(exp), f'{sim.shape} {exp.shape}'
    if log:
        sim = np.log(sim + 1)
        exp = np.log(exp + 1)

    npixels = np.shape(sim)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)

    # make composite contact map
    composite = np.zeros((npixels, npixels))
    composite[indu] = sim[indu]
    composite[indl] = exp[indl]

    if vmaxp is None:
        vmaxp = np.mean(exp)

    plot_matrix(composite, ofile, title, vmax = vmaxp, triu = True, cmap = cmap)


def plot_tri(first, second, vmaxp=None, oe=False, title="", dark=False, log=False):
    """compare two contact maps in upper and lower triangles"""
    if dark:
        vmaxp = np.mean(first) / 2
        absolute = True

    assert np.shape(first) == np.shape(second)

    if oe:
        plot_fn = plot_oe
        first = get_oe(first)
        second = get_oe(second)
    else:
        plot_fn = plot_contactmap

    composite = make_tri_composite(first, second)
    symmetry_score = get_symmetry_score(composite)

    if vmaxp is None:
        plot_fn(composite, title=title, log=log)
    else:
        plot_fn(
            composite,
            vmaxp=vmaxp,  # pyright: ignore
            absolute=True,  # pyright: ignore
            title=title,
            log=log,
        )

    plt.title("symmetry score: {%.2f}" % (symmetry_score))


def plot_obs_vs_goal(sim):
    """compare observables versus maxent goals"""
    if sim.obs_tot is None:
        return
    fig, axs = plt.subplots(2, figsize=(12, 14))
    obs = sim.obs_tot
    goal = sim.obj_goal_tot
    axs[0].plot(obs, "--o", label="obs")
    axs[0].plot(goal, "ko", label="goal")
    if sim.obs is not None:
        axs[0].vlines(len(sim.obs), min(sim.obs_tot) / 5, max(sim.obs_tot) / 5, "k")
    axs[0].legend()
    axs[0].set_title("Observables vs Goals")

    diff = sim.obs_tot - sim.obj_goal_tot
    axs[1].plot(diff, "--o")
    axs[1].hlines(0, len(sim.obs_tot), 0, "k")
    if sim.obs is not None:
        axs[1].vlines(len(sim.obs), min(diff) / 5, max(diff) / 5, "k")
    axs[1].set_title("Difference")


class parameters:
    def __init__(self, N):
        self.N = N
        self.baseN = 512000
        self.basev = 520
        self.baseb = 16.5
        self.side_length = 1643  # nm
        self.frac = self.baseN / self.N

        self.v = self.basev * self.frac
        self.b = self.baseb * np.sqrt(self.frac)

    def convertv(self, N):
        return self.basev * self.frac

    def convertb(self, N):
        return self.baseb * np.sqrt(self.frac)

    def delta(self, nint):
        self.nsite = self.N / nint
        self.L = self.nsite ** (1 / 3)
        return self.side_length / self.L

    def nint(self, delta):
        L = self.side_length / delta
        nsites = L**3
        nint = self.N / nsites
        return nint


def load_contactmap_hicstraw(hicfile, res, chrom, start, end, clean=False, KR=True):
    assert res in hicfile.getResolutions()

    if KR:
        mzd = hicfile.getMatrixZoomData(chrom, chrom, "observed", "KR", "BP", res)
    else:
        mzd = hicfile.getMatrixZoomData(chrom, chrom, "observed", "NONE", "BP", res)

    if res > 50000:
        contact = mzd.getRecordsAsMatrix(start, end, start, end)
    else:
        contact = load_contactmap_with_buffers(mzd, start, end, res)

    if clean:
        contact, dropped_indices = clean_contactmap(contact)

    return contact


def load_contactmap_with_buffers(mzd, start, end, res):
    """have to buffer getRecordsAsMatrix otherwise bad things happen"""
    bufsize = 5_000_000  # increment
    width = end - start

    print(width)
    assert width % res == 0
    assert width % bufsize == 0

    pixels = int(width / res)
    steps = int(width / bufsize)

    assert pixels % steps == 0
    pix_per_buf = int(pixels / steps)
    print(pixels)

    out = np.zeros((pixels, pixels))

    logging.info("loading contact map with buffers")
    for i in tqdm(range(steps)):
        for j in range(steps):
            xmin = i * bufsize
            xmax = (i + 1) * bufsize
            ymin = j * bufsize
            ymax = (j + 1) * bufsize

            imin = i * pix_per_buf
            imax = (i + 1) * pix_per_buf
            jmin = j * pix_per_buf
            jmax = (j + 1) * pix_per_buf

            buffer = mzd.getRecordsAsMatrix(xmin, xmax - 1, ymin, ymax - 1)
            out[imin:imax, jmin:jmax] = buffer

    return out


def make_clean_mask(inds, N):
    mask = np.full((N, N), True)
    for i in inds:
        mask[i, :] = False
        mask[:, i] = False

    return mask

def drop_row_col(chis, inds):
    N = len(chis)
    mask = np.full((N, N), True)
    for i in inds:
        mask[i, :] = False
        mask[:, i] = False

    deleted = len(inds)
    return chis[mask].reshape(N - deleted, N - deleted)

def clean_contactmap(contact):
    """if the main diagonal entry is zero, remove the entire row and column"""
    N, _ = np.shape(contact)
    d = np.diagonal(contact)
    inds = np.where(d == 0)[0]
    mask = make_clean_mask(inds, N)
    deleted = len(inds)

    return contact[mask].reshape(N - deleted, N - deleted), inds
