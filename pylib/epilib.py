import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import skimage
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import sklearn.metrics
#import straw
#import strawC
from os import system
import os
import os.path as osp
import copy
import json
from scipy.sparse import coo_matrix
from numba import njit
from tqdm import tqdm
#import kmeans

#import palettable
#from palettable.colorbrewer.sequential import Reds_3


from pylib import utils

mycmap = mpl.colors.LinearSegmentedColormap.from_list('custom',
                                             [(0,    'white'),
                                              (0.3,  'white'),
                                              (1,    '#ff0000')], N=126)


#%config InlineBackend.figure_format='retina'
#plt.rcParams['figure.figsize'] = [8,6]
#plt.rcParams.update({'font.size':14})    
        

class Sim:
    
    def __init__(self):
        print("you're in here")

    def __init__(self, path):
        self.path = path
        self.metrics = {}
        
        try:
            self.config = self.load_config()
        except:
            print("no config.json")
        
        self.diag_bins = np.shape(self.config['diag_chis'])[0]
        self.chi = self.load_chis()

        try:
            self.hic = get_contactmap(osp.join(path,"contacts.txt"))
            self.raw = get_contactmap(osp.join(path,"contacts.txt"), rawcounts=True)
            self.d = get_diagonal(self.hic)
        except:
            print("error loading contactmap.")
        
        try:
            self.energy = pd.read_csv(path+"/energy.traj", sep='\t', names=["step", "bonded", "nonbonded", "diagonal", "y"])
        except:
            print("no energy.traj") 
        
        try:
            self.seqs = self.load_seqs()
            self.k = np.shape(self.seqs)[0]
        except:
            print("no seqs")
            
        try:
            self.obs_full = pd.read_csv(osp.join(path,"observables.traj"), sep ='\t', header=None)
            self.obs = self.obs_full.mean().values[1:]
        except:
            print("no plaid observables")
        
        try:
            self.diag_obs_full = pd.read_csv(osp.join(path,"diag_observables.traj"), sep ='\t', header=None)
            self.diag_obs = self.diag_obs_full.mean().values[1:]
        except:
            print("no diag observables")
        
        try:
            self.obs_tot = np.hstack((self.obs, self.diag_obs))
            self.extra = np.loadtxt(osp.join(path, "extra.traj"))
            self.beadvol = self.config["beadvol"]
            self.nbeads = self.config["nbeads"]
        except:
            print("something wrong")
            
        resources_path = osp.join(path, "../../resources/")
        if os.path.exists(resources_path):
            self.resources_path = resources_path
        
        gthic_possibilites = [".", "..", "../../resources"]
        gthic_loaded = False
        for gtp in gthic_possibilites:
            gthic_path = osp.join(self.path, gtp, "experimental_hic.npy")
            print("looking in, ", gthic_path)
            if os.path.exists(gthic_path) and not gthic_loaded:
                self.gthic = np.load(gthic_path)
                gthic_loaded = True
                
        if not gthic_loaded:
            print("no ground truth hic found")
            

        
        obj_goal_path = osp.join(resources_path, "obj_goal.txt")
        if os.path.exists(obj_goal_path):
            self.obj_goal = np.loadtxt(obj_goal_path)
        else:
            print("no path to obj_goal.txt")
            print("looking in obj_goal_path: ", obj_goal_path)
        
        obj_goal_diag_path = osp.join(resources_path, "obj_goal_diag.txt")
        if os.path.exists(obj_goal_diag_path):
            self.obj_goal_diag = np.loadtxt(obj_goal_diag_path)
        else:
            print("no path to obj_goal_diag.txt")
            print("looking in obj_goal_diag_path: ", obj_goal_path)
        
        try:
            self.obj_goal_tot = np.hstack((self.obj_goal, self.obj_goal_diag))
        except:
            try:
                params_path = osp.join(resources_path, "params.json")
                params = utils.load_json(params_path)
                self.obj_goal_tot = params["goals"]
                print("looking for goals in params")
            except:
                print("no params found")
           
    def init_sim(self, overwrite=True, getgoals=False):
        if os.path.exists(self.path):
            if not overwrite:
                print("path already exists, exiting. to overwrite, set overwrite=True")
                return 
        else:
            os.makedirs(self.path)
        
        self.save_chis()
        k = len(self.seqs)

        with open(osp.join(self.path, "config.json"), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        if not os.path.exists(osp.join(self.path, "TICG-engine")):
            os.system("ln -s ~/Documents/TICG-chromatin/src/TICG-engine " + str(self.path))
        
        self.config['nbeads'] = np.shape(self.gthic)[0]
                
        if getgoals:
            ndiag_bins = 32
            nchis =  k*(k+1)/2
            goals_plaid = get_goal_plaid(self.gthic, self.seqs, self.config)
            goals_diag = get_goal_diag(self.gthic, self.config, ndiag_bins=ndiag_bins, dense_diagonal_on=True)

            np.savetxt(osp.join(self.path,"chis.txt"), np.zeros((2,nchis)), fmt="%.8f")
            np.savetxt(osp.join(self.path,"chis_diag.txt"), np.zeros((2,ndiag_bins)), fmt="%.8f")

            np.savetxt(osp.join(self.path,"obj_goal.txt"), goals_plaid, newline=" ", fmt="%.8f")
            np.savetxt(osp.join(self.path,"obj_goal_diag.txt"), goals_diag, newline=" ", fmt="%.8f")

        # TODO: also save bead type names into config, if different.
        for i, seq in enumerate(self.seqs):
            pcfname = "pcf"+str(i+1)+".txt"
            #chipseq_files.append(pcfname)
            np.savetxt(osp.join(self.path, pcfname), seq, newline="\n", fmt="%.8f")
            
        np.save(osp.join(self.path,"experimental_hic.npy"), self.gthic)


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
        with open(self.path + "/config.json") as f:
            config = json.load(f)
            
        return config
    
    def load_chis(self):
        try:
            nspecies = self.config['nspecies']
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

            chi = np.zeros((nspecies, nspecies))
            for i in range(nspecies):
                for j in range(nspecies):
                    if j >= i:
                        chi[i,j] = self.config["chi" + letters[i] + letters[j]]
                        chi[j,i] = self.config["chi" + letters[i] + letters[j]]
        except KeyError:
            indices = np.triu_indices(self.config["nspecies"])
            chi = np.array(self.config["chis"])[indices]
            
        return chi
    
    def save_chis(self):
        nspecies = self.config['nspecies']
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        chi = np.zeros((nspecies, nspecies))
        for i in range(nspecies):
            for j in range(nspecies):
                if j >= i:
                    self.config["chi" + letters[i] + letters[j]] = self.chi[i,j]
                    self.config["chi" + letters[i] + letters[j]] = self.chi[j,i] 
 
    def load_seqs(self):
        seqs = []
        for file in self.config["bead_type_files"]:
            seqs.append( np.loadtxt(self.path + "/" + file) )
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
        smatrix = seqs.T @ chimatrix @ seqs
        
        return smatrix
    
    def plot_contactmap(self, vmaxp=0.1, absolute=False, dark=False, title="",log=False):
        plot_contactmap(self.hic, vmaxp, absolute, dark=dark, title=title)
    
    def plot_diag(self, scale="semilogy", label=None):
        plot_diag(self.hic, scale, label=label)
    
    def plot_energy(self):
        plot_energy(self)
    
    def plot_obs(self, diag):
        plot_obs(self, diag)
        
    def plot_tri(self, vmaxp=None, title="", dark=False, log=False):
        plot_tri(self.hic, self.gthic, vmaxp, title=title, dark=dark, log=log)
        
    def plot_scatter(self):
        plot_scatter(self.hic, self.gthic)
        
    def plot_diff(self):
        plot_diff(self.hic, self.gthic)
  
    '''
    def get_goal_plaid(self, flat=True):
        k = self.config["nspecies"]
        goal_exp = np.zeros((k,k))
        
        for i, seqi in enumerate(self.seqs):
            for j, seqj in enumerate(self.seqs):
                goal_exp[i,j] = np.mean((np.outer(seqi,seqj)*self.hic).flatten())
        
        goal_sim = np.zeros((k,k))
        goal_sim[0,0] = np.mean(self.obs[1])
        goal_sim[1,0] = np.mean(self.obs[2])
        goal_sim[0,1] = np.mean(self.obs[2])
        goal_sim[1,1] = np.mean(self.obs[3])
        
        if flat:
            ind = np.triu_indices(k)
            goal_sim = goal_sim[ind]        
        
        return goal_exp, goal_sim
    '''
    '''
    def get_goal_diag(self):
        nbeads = self.config['nbeads']
        bins = len(self.config['diag_chis'])
        diag_mask, correction = mask_diagonal(self.hic, bins, nbeads)
        return diag_mask
        '''

    def process(self):
        diag_sim = np.mean(self.diag_obs)[1:]
        diag_sim = np.array(diag_sim)

        nbeads = self.config['nbeads']
        bins = len(self.config['diag_chis'])
        assert(bins == len(diag_sim))
        
        diag_exp = downsample(self.d, int(nbeads/bins))
        diag_mask, correction = mask_diagonal(self.hic, bins)
        
        return diag_sim, diag_exp, diag_mask, correction

    def plot_obs_vs_goal(self):
        plot_obs_vs_goal(self)
    
    def plot_oe(self,log=False):
        plot_oe(get_oe(self.hic))
        
    def plot_consistency(self):
        return plot_consistency(self)
    
    def calc_goals(self):
        plaid = get_goal_plaid(self.gthic, self.seqs,  self.config)
        diag = get_goal_diag(self.gthic, self.config, ndiag_bins=self.diag_bins, adj=True, dense_diagonal_on=self.config['dense_diagonal_on'])
   
        np.savetxt("obj_goal.txt", plaid, newline=" ", fmt="%.8f")
        np.savetxt("obj_goal_diag.txt", diag, newline=" ", fmt="%.8f")
    
    def plot_diagonal(self, *args, scale="semilogy"):
        
        if scale == "semilogy":
            plot_fn = plt.semilogy
        elif scale == "log" or scale == "loglog":
            plot_fn = plt.loglog
        
        diag = self.d
        plot_fn(np.linspace(1/len(diag),1,len(diag)), diag, *args, label="sim")
        try:
            diag = get_diagonal(self.gthic)
            plot_fn(np.linspace(1/len(diag),1,len(diag)), diag, "k", label="exp")
        except:
            print("no ground truth hi-c for plot_diagonal")

        plt.xlabel("genomic distance")
        plt.ylabel("probability")
        plt.legend()
        plt.title("probability vs genomic distance")
        lower_bound = np.max([np.min(diag)/5, 1e-7])
        plt.axis([0,1,lower_bound, 1])
        
class SCC():
    '''
    Class for calculation of SCC as defined by https://pubmed.ncbi.nlm.nih.gov/28855260/
    '''
    def __init__(self):
        self.r_2k_dict = {} # memoized solution for var_stabilized r_2k
    def r_2k(self, x_k, y_k, var_stabilized):
        '''
        Compute r_2k (numerator of pearson correlation)
        Inputs:
            x: contact map
            y: contact map of same shape as x
            var_stabilized: True to use var_stabilized version
        '''
        # x and y are stratums
        if var_stabilized:
            N_k = len(x_k)
            if N_k in self.r_2k_dict:
                result = self.r_2k_dict[N_k]
            else:
                result = np.var(np.arange(1, N_k+1)/N_k)
                self.r_2k_dict[N_k] = result
            return result
        else:
            return np.sqrt(np.var(x_k) * np.var(y_k))
    def mean_filter(x, size):
        return scipy.ndimage.uniform_filter(x, size, mode = 'constant') / (size)**2
    def scc(self, x, y, h = 3, K = None, var_stabilized = False, verbose = False):
        '''
        Compute scc between contact map x and y.
        Inputs:
            x: contact map
            y: contact map of same shape as x
            h: span of convolutional kernel (width = (1+2h))
            K: maximum stratum (diagonal) to consider (None for all)
            var_stabilized: True to use var_stabilized r_2k
            verbose: True to print when nan found
        '''
        x = SCC.mean_filter(x.astype(np.float64), 1+2*h)
        y = SCC.mean_filter(y.astype(np.float64), 1+2*h)
        if K is None:
            K = len(y) - 2
        num = 0
        denom = 0
        nan_list = []
        for k in range(1, K):
            # get stratum (diagonal) of contact map
            x_k = np.diagonal(x, k)
            y_k = np.diagonal(y, k)
            N_k = len(x_k)
            r_2k = self.r_2k(x_k, y_k, var_stabilized)
            #p_k, _ = pearsonr(x_k, y_k)
            p_k = np.corrcoef(x_k, y_k)[0,1]
            if np.isnan(p_k):
                # nan is hopefully rare so just set to 0
                nan_list.append(k)
                p_k = 0
                if verbose:
                    print(f'k={k}')
                    print(x_k)
                    print(y_k)
            num += N_k * r_2k * p_k
            denom += N_k * r_2k
        if len(nan_list) > 0:
            print(f'{len(nan_list)} nans: k = {nan_list}')
        return num / denom           

def get_contactmap(filename, norm=True, log=False, rawcounts=False, normtype="max"):
    """Loads contact map from file, returns array."""
    
    if type(filename) is str:
        contactmap = np.loadtxt(filename)
    elif type(filename) is np.ndarray:
        contactmap = copy.deepcopy(filename)
    else:
        print("usage: get_contactmap filename:[str, np.ndarray], norm:bool, log:bool, rawcounts:bool")
        raise ValueError
    
    if rawcounts:     
        return contactmap
    else:  
        if norm:
            # rescale_contactmap(contactmap, method=normtype)
            if normtype == "max":
                contactmap /= np.max(np.diagonal(contactmap))
            elif normtype == "mean":
                contactmap /= np.mean(np.diagonal(contactmap))
                np.fill_diagonal(contactmap, 1)
            #df /= df[0][0]
        if log:
            contactmap = np.log(contactmap)
    
        return contactmap

def rescale_contactmap(contactmap, method="mean"):
    """rescale contact map so that the entries are probabilities rather than frequencies"""
    if method == "max":
        contactmap /= np.max(np.diagonal(contactmap))
    elif method == "mean":
        contactmap /= np.mean(np.diagonal(contactmap))
        np.fill_diagonal(contactmap, 1)
    return contactmap

'''
def get_contactmap_matrix(contactmap, norm=True, log=False):
    """Loads contact map from contacts.txt matrix, returns array."""
    if norm:
        contactmap /= max(np.array(contactmap).diagonal())
        #df /= df[0][0]

    if log:
        contactmap = np.log(contactmap)
        
    return contactmap
'''

@njit
def get_diagonal(contact):
    """Returns the probablity of contact as a function of genomic distance"""
    rows, cols = np.shape(contact)
    d = np.zeros(rows)
    for k in range(rows):
        d[k] = np.mean(np.diag(contact, k))
    return d

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
                oe[i,j] = 1
            else:      
                oe[i,j] = contact[i,j]/d
        
    return oe

def plot_oe(oe, title="", log = False):
    plt.figure(figsize=(12,10))
    plt.imshow(oe, vmin=0, vmax=2, cmap='bwr')
    plt.title(title)
    plt.colorbar()

def plot_contactmap(contact, vmaxp=0.1, absolute=False, imshow=True, cbar=True, dark=False, title="", log=False):
    plt.figure(figsize=(12,10))
    
    cmap = mycmap
    vmin = 0
    
    if dark:
        vmaxp = np.mean(contact)/2
        absolute = True
        
    if absolute:
        vmax = vmaxp
    else:
        vmax = contact.mean()+vmaxp*contact.std()
    
    if imshow:
        plot_fn = plt.imshow
    else:
        plot_fn = sns.heatmap
        
    if log:
        contact = np.log10(contact+1e-20)
        vmin = np.min(contact[ np.where(contact > -19) ])
        vmax = vmin / 3
        print("vmiin", vmin)
        print("vmax", vmax)
        
    plot_fn(contact, cmap=cmap, vmin=vmin, vmax=vmax)
    
    if cbar:
        plt.colorbar()
        plt.title(title)

def plot_diagonal(diag, *args, scale="semilogy", label=None):
    if scale=="semilogy":
        plt.semilogy(np.linspace(1/len(diag),1,len(diag)), diag, *args, label=label)
    elif scale=="loglog":
        plt.loglog(np.linspace(1/len(diag),1,len(diag)), diag, *args, label=label)
        
        
def plot_energy(sim):
    fig, axs = plt.subplots(3,1, figsize=(12,10))
    sz = np.shape(sim.energy)[0]
    last20 = int(sz - sz/5)
    
    bondmean = np.mean(sim.energy['bonded'][:last20])
    nbondmean = np.mean(sim.energy['nonbonded'][:last20])
    diagmean = np.mean(sim.energy['diagonal'][:last20])
    
    xaxis = np.array(sim.energy['step'])

    axs[0].plot(xaxis, sim.energy['bonded'], label='bonded')
    axs[1].plot(xaxis, sim.energy['nonbonded'], label='nonbonded')
    axs[2].plot(xaxis, sim.energy['diagonal'], label='diagonal')
    axs[0].hlines(bondmean, 0, xaxis[-1], colors='k')
    axs[1].hlines(nbondmean, 0, xaxis[-1], colors='k')
    axs[2].hlines(diagmean, 0, xaxis[-1], colors='k')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    
def plot_obs(sim, diag=True):
    o = np.array(sim.obs_full.T)
    d = np.array(sim.diag_obs_full.T)
    plt.plot(o[1:].T);
    if diag:
        plt.figure()
        plt.semilogy(d[1:].T);

def plot_diff(first, second, c=None, log=False, oe=False, title=""):
    if c is None and log is False:
        c = first.mean() + 0.1*first.std()
    
    if log:
        print("log is true")
        frac = np.log(first+1e-20), np.log(second+1e-20)
        plt.imshow(frac)
        return
        
    plt.figure(figsize=(12,10))
    diff = first-second
    
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
    plt.xlabel("Pearson: {%.4f}, SCC: {%.4f}, RMSE: {%.4f}" %(pearson, scc, rmse))
            
    plt.imshow(diff, vmin=vmin, vmax=vmax, cmap='bwr')
    plt.colorbar()
    
    return scc
    
def randomized_svd(X, r, q=10, p=1):
    """randomized svd for decomposing large matrices
    X: ndarray to decompose
    r: number of singular values
    q: oversampling
    p: power iterations"""
    ny = X.shape[1]
    P = np.random.randn(ny, r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z, mode="reduced")
    
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    
    return U, S, VT
    
def get_sequences(hic, k, dtype="float", map01=False, map11=True, split=False, randomized=False, scaleby_singular_values=False, scaleby_sqrt_singular_values=False):
    """
    calculate polymer bead sequences using k principal components
    returns 2*k epigenetic sequencess

    scaleby_singular_values: (bool): scales principal components by their singular value
    """
    OEmap = get_oe(hic)
    if randomized:
        print("getting sequences with RANDOMIZED SVD")
        U, S, VT = randomized_svd(np.corrcoef(OEmap), 2*k)
    else:    
        print("getting sequences with np.linalg.svd")
        U, S, VT = np.linalg.svd(np.corrcoef(OEmap), full_matrices=0)
    
    pcs = []
    for i in range(k):
        if split:
            pcs.append(np.array([positive(n) for n in VT[i]]))
            pcs.append(-np.array([negative(n) for n in VT[i]]))
        else:
            pcs.append(np.array(VT[i]))
            if map11:
                for i, pc in enumerate(pcs):
                    pcs[i] = pc/max(abs(pc))
    
    if scaleby_singular_values:
        assert(split == False)
        for i,pc in enumerate(pcs):
            pc *= S[i]

    if scaleby_sqrt_singular_values:
        assert(split == False)
        for i,pc in enumerate(pcs):
            pc *= np.sqrt(S[i])
    
    assert(dtype=="int" or dtype=="float")
    
    beads_per_bin = 1 
    seqs = []
    for pc in pcs:
        seqs.append(seq_from_pc(pc, beads_per_bin, dtype, map01))
    
    return np.array(seqs)

def map_0_1(signal, backfrom=0):
    """Returns signal data mapped to the interval [0,1]"""
    signal = signal.real
    #signal = (signal - np.min(signal[:-backfrom]))/(np.max(signal) - np.min(signal[:-backfrom]))
    signal = (signal - np.min(signal))/(np.max(signal) - np.min(signal))

    #signal[signal<0] = 0
    return signal

def map_0_1_chip(signal, backfrom=0):
    """Returns signal data mapped to the interval [0,1]"""
    signal = signal.real
    signal = (signal - np.min(signal[:-backfrom]))/(np.max(signal) - np.min(signal[:-backfrom]))
    signal[signal<0] = 0
    signal *= 1.5
    signal[signal>1] = 1
    return signal

def get_baseline_signal(signal):
    bin_populations, bin_signal = np.histogram(signal, bins=100)
    baseline_signal = bin_signal[np.argmax(bin_populations)]
    return baseline_signal

def new_map_0_1_chip(signal):
    baseline  = get_baseline_signal(signal)
    signal = (signal - baseline)/(np.max(signal) - baseline)
    signal[signal<0] = 0
    signal *= 1.5
    signal[signal>1] = 1
    return signal

def check_baseline(chipseq):
    """ plots chipseq and the calculalted baseline signal"""
    baseline_signal = get_baseline_signal(chipseq)
    plt.plot(chipseq)
    plt.hlines(baseline_signal, 0, 1024, 'r')
    print(baseline_signal)


def seq_from_pc(pc, beads_per_bin = 1, dtype="int", map01=True):
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
    """returns f(x) < 0 ? 0 """
    if f<0:
        return 0
    else:
        return f

def negative(f):
    """returns f(x) > 0 ? 0 """
    if f>0:
        return 0
    else:
        return f

def plot_smooth(data, n_steps=20, llabel="none", fmt='o', norm=True):
    """Returns data smoothed in a window of size n_steps."""
    time_series_df = pd.DataFrame(data)
    smooth_path    = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line     = (smooth_path-path_deviation)[0]
    over_line      = (smooth_path+path_deviation)[0]

    #Plotting:
    if norm:
        s = np.array(range(len(data)))/len(data)
    else:
        s = np.array(range(len(data)))
    plt.plot(s, smooth_path, fmt, linewidth=2, label=llabel) #mean curve.
    plt.fill_between(s, under_line, over_line, color='b', alpha=.1) #std curves


def half_sample(contact):
    rows, cols = contact.shape
    i_vals = np.arange(0,int(rows/2),2)
    j_vals = np.arange(0,int(cols/2),2)

    x = np.zeros((int(rows/2), int(cols/2)))

    for i in range(int(rows/2)):
        for j in range(int(cols/2)):
            x[i][j] = 0.25*(contact[i*2][j*2] + contact[i*2+1][j*2] + contact[i*2][j*2+1] + contact[i*2+1][j*2+1])

    return x

def process(folder, ignore=100):
    data_out_filename = "data_out"
    contact = get_contactmap("./" + folder + "/" + data_out_filename + "/contacts.txt")
    #seq1 = np.array(pd.read_csv(folder + "/seq1.txt"))
    #seq2 = np.array(pd.read_csv(folder + "/seq2.txt"))
    seq1 = np.array(pd.read_csv(folder + "/pc1.txt"))
    seq2 = np.array(pd.read_csv(folder + "/pc2.txt"))

    goal_exp = np.zeros((2,2))
    goal_exp[0,0] = np.mean((np.outer(seq1,seq1)*contact).flatten())
    goal_exp[1,0] = np.mean((np.outer(seq1,seq2)*contact).flatten())
    goal_exp[1,1] = np.mean((np.outer(seq2,seq2)*contact).flatten())

    obs = pd.read_csv("./" + folder + "/" + data_out_filename + "/observables.traj", sep='\t', header=None)
    goal_sim = np.zeros((2,2))
    
    goal_sim[0,0] = np.mean(obs[1][ignore:])
    goal_sim[1,0] = np.mean(obs[2][ignore:])
    goal_sim[1,1] = np.mean(obs[3][ignore:])

    print("goals from experiment")
    print("calculated as weighted average from contact map")
    print(goal_exp)
    print("goals from simulation")
    print("calculated as mean of observables")
    print(goal_sim)
    
def process2(folder, ignore=100):
    contact = get_contactmap("./" + folder + "/contacts.txt")
    
    seqs = []
    filenames = os.popen("ls "+folder+"/pc*").read().split("\n")[:-1]
    print(filenames)
    for file in filenames:
        seqs.append(np.loadtxt(file))
        
    k = len(seqs)
    goal_exp = np.zeros((k,k))
    for i, seqi in enumerate(seqs):
        for j, seqj in enumerate(seqs):
            goal_exp[i,j] = np.mean((np.outer(seqi,seqj)*contact).flatten())

    obs = pd.read_csv("./" + folder + "/observables.traj", sep='\t', header=None)
 
    # remove first column corresponding to MC iteration
    goal_sim = np.zeros((k,k))

    #for i in range(k):
        #for j in range(i, k):
            #goal_sim[i,j] = np.mean(obs[1 + i*k + j][ignore:])
            
    goal_sim[0,0] = np.mean(obs[1])
    goal_sim[1,0] = np.mean(obs[2])
    goal_sim[0,1] = np.mean(obs[2])
    goal_sim[1,1] = np.mean(obs[3])


    print("goals from experiment")
    print("calculated as weighted average from contact map")
    print(goal_exp)
    print("goals from simulation")
    print("calculated as mean of observables")
    print(goal_sim)

def get_diag_obs(filename):
    goal_sim[0,0] = np.mean(obs[1][ignorefirst:])
    goal_sim[1,0] = np.mean(obs[2][ignorefirst:])
    goal_sim[1,1] = np.mean(obs[3][ignorefirst:])

    print("goals from experiment:")
    print(goal_exp)
    print("goals from simulation:")
    print(goal_sim)

def compare_diagonal(filename, gridsize, plot=True, getgoal=True):
    df = pd.read_csv(filename + "/data_out/diag_observables.traj", sep="\t", header=None)
    diag_sim = df.mean()[1:]
    diag_sim = np.array(diag_sim)
    return diag_sim

def mask_vs_diagonal(contact, nbeads=1024, bins=16):
    diag_exp = get_diagonal(contact)
    diag_exp = downsample(diag_exp, int(nbeads/bins))
    
    diag_mask, correction = mask_diagonal(contact, bins)
    
    plt.plot(np.log10(diag_exp) ,'--o')
    plt.plot(np.log10(diag_mask/correction), '--o')

def compare_diagonal(filename, nbeads=1024, bins=16, plot=True, getgoal=False, fudge_factor=1):
    print("diag_sim is volume fraction, average of diag_observables.traj")
    print("diag_exp is p(s), average of each subdiagonal which is then downsapled to match resolution of diagonal bins")
    print("diag_mask is mask, weighted average of contact map")
    print("corrected is mask, with correction factor to match p(s)")
    df = pd.read_csv(filename + "/diag_observables.traj", sep="\t", header=None)
    diag_sim = df.mean()[1:]
    diag_sim = np.array(diag_sim)

    if getgoal:
        df_goal = pd.read_csv(filename + "../../obj_goal_diag.txt", header=None, sep=" ")
        diag_sim_goal = df_goal.mean()[1:]
        diag_sim_goal = np.array(diag_sim_goal)

    diag_std = df.std()[1:]
    diag_std = np.array(diag_std)
    #plt.plot(diag_sim, 'o')

    contact = get_contactmap(filename + "/contacts.txt")
    diag_exp = get_diagonal(contact)
    diag_exp = downsample(diag_exp, int(nbeads/bins))

    diag_mask, correction = mask_diagonal(contact, bins)

    #print(np.mean(diag_sim/diag_mask))

    #plt.errorbar(np.asarray(range(len(diag_sim))), np.log10(diag_sim), diag_std, '--o', label="sim")
    #plt.plot(np.log10(diag_sim/correction/22), '--o', label="volume_fraction")
    if plot:
        plt.figure(figsize=(12,10))
        plt.plot(np.log10(diag_sim), '--o', label="volume_fraction")
        plt.plot(np.log10(diag_exp), '--o', label="p(s)")
        plt.plot(np.log10(diag_mask* fudge_factor), '--o', label="mask")
        #plt.plot(np.log10(diag_mask), '--o', label="mask")
        plt.plot(np.log10(diag_mask/correction), '--o', label="corrected")
        
        plt.plot(np.log10(diag_sim/fudge_factor/correction), '--o', label="volfrac converted")
        if getgoal:
            plt.plot(np.log10(diag_sim_goal), label="vol frac goal")
        plt.legend()
        plt.xlabel('s')
        plt.ylabel('probability')

    #print(diag_sim_goal)
    #print(diag_sim/diag_mask)
    print(np.mean(diag_sim/diag_mask))

    return diag_sim, diag_exp, diag_mask, correction

@njit
def make_mask(size, b, cutoff, loading, ndiag_bins, dense_diagonal_on):
    """ makes a mask with 1's in subdiagonals inside
    
    actually faster than numpy version when jitted
    """
    
    rows, cols = size, size
    mask = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            #if int((r-c)/binsize) == b:
            bin_index = binDiagonal(r, c, cutoff, loading, ndiag_bins, rows, dense_diagonal_on)
            if bin_index == b:
                mask[r,c] = 1
                mask[c,r] = 1
                if r==c:
                    mask[r,r] = 2
    return mask

@njit
def make_mask_fast(size, binsize, b):
    # not actually faster if you jit the make_mask code
    rows, cols = size, size
    mask = np.zeros((rows, cols))
    for i in range(binsize):
        mask += np.eye(rows, cols, b*binsize+i)
        mask += np.eye(rows, cols, -b*binsize-i)
    return mask

@njit
def mask_diagonal(contact, cutoff, loading, ndiag_bins, dense_diagonal_on):
    """Returns weighted averages of contact map"""
    rows, cols = contact.shape
    #binsize = int(rows/ndiag_bins)
    
    assert(rows==cols), "contact map must be square"
    nbeads = rows
    measure = []
    correction = []

    for b in range(ndiag_bins):
        '''
        mask = np.zeros_like(contact)
        for r in range(rows):
            for c in range(cols):
                if int((r-c)/binsize) == b:
                    mask[r,c] = 1
                    mask[c,r] = 1
                    if r==c:
                        mask[r,r] = 2
        '''
        mask = make_mask(nbeads, b, cutoff, loading, ndiag_bins, dense_diagonal_on)
        #measure.append(np.mean((mask*contact).flatten()))
        #correction.append(np.sum(mask)/nbeads**2)
        measure.append(np.sum((mask*contact).flatten()))        
        correction.append(np.sum(mask))
    '''

    for b in range(bins):            
        mask = make_mask_fast(rows, cols, binsize, b)
        measure.append(np.mean((mask*contact).flatten()))
        correction.append(np.sum(mask)/nbeads**2)
    '''

    
    measure = np.array(measure)
    correction = np.array(correction)
    return measure, correction

@njit
def binDiagonal(i, j, cutoff, loading, ndiag_bins, nbeads, dense_diagonal_on):
    s = abs(i-j)

    if dense_diagonal_on:
        #loading = config["loading"]
        #cutoff = config["cutoff"]
        dividing_line = nbeads*cutoff

        n_small_bins = int(loading*ndiag_bins)
        n_big_bins = ndiag_bins-n_small_bins
        small_binsize = int(dividing_line/(n_small_bins))
        big_binsize = int((nbeads-dividing_line)/n_big_bins)

        if s > dividing_line:
            bin_index = n_small_bins + np.floor( (s - dividing_line) / big_binsize)
        else:
            bin_index =  np.floor( s / small_binsize)
    else:
        binsize = nbeads/ndiag_bins
        bin_index = np.floor( s / binsize )
    
    return bin_index
                   
def downsample(sequence, res):
    """
    take sequence of numbers and reduce length
    res: new step size
    """
    #assert(len(sequence)%res == 0)
    new = []
    for i in range(0, len(sequence), res):
        new.append(np.mean(sequence[i:i+res]))

    return np.array(new)

def get_goal_plaid(hic, seqs, config, flat=True, norm=False, adj=True):
    """
    flat: return vector. else return matrix of chis.
    """
    k, n = np.shape(seqs)
    goal_exp = np.zeros((k,k))
    for i, seqi in tqdm(enumerate(seqs)):
        for j, seqj in enumerate(seqs):
            #goal_exp[i,j] = np.mean((np.outer(seqi,seqj)*hic).flatten())
            goal_exp[i,j] = np.sum((np.outer(seqi,seqj)*hic).flatten())
            
            if adj:
                vbead = config["beadvol"]
                vcell = config["grid_size"]**3
                goal_exp[i,j] *= vbead/vcell
                
            if norm == "abs":
                #goal_exp[i,j] /= np.sum(np.outer(np.abs(seqi), np.abs(seqj)))/np.shape(hic)[0]**2
                goal_exp[i,j] /= np.sum(np.outer(np.abs(seqi), np.abs(seqj)))
                
            if norm == "n2":
                goal_exp[i,j] /= np.shape(hic)[0]**2
            
            if norm == "n":
                goal_exp[i,j] /= np.shape(hic)[0]
            
            if norm == "nlogn":
                n = np.shape(hic)[0]
                goal_exp[i,j] /= n * np.log10(n)
            
            if norm == "n1.5":
                goal_exp[i,j] /= np.shape(hic)[0]**1.5
                
                            
            if norm == "nsqrt":
                n = np.shape(hic)[0]
                f = np.sum(np.outer(np.abs(seqi), np.abs(seqj)))
                goal_exp[i,j] /= f / np.sqrt(n)
                
    if flat:
        ind = np.triu_indices(k)
        goal_exp = goal_exp[ind]
        
    return goal_exp

def get_goal_plaid2(hic, seqs, k, flat=True):
    """
    do we need to get the correct denominator?? 
    k is the number of sequences (not pcs)
    flat: return vector. else return matrix of chis.
    """
   
    goal_exp = np.zeros((k,k))
    for i, seqi in enumerate(seqs):
        for j, seqj in enumerate(seqs):
            goal_exp[i,j] = np.mean((np.outer(seqi,seqj)*hic).flatten())
            # correction = np.sum(np.outer(seqi,seqj)) / nbeads**2

    if flat:
        ind = np.triu_indices(k)
        goal_exp = goal_exp[ind]
        
    return goal_exp

def get_goal_diag(hic, config, ndiag_bins=32, getcorrect=False, adj=True, dense_diagonal_on=True):
    # TODO - get dense diagonal on from config
    cutoff = config["dense_diagonal_cutoff"]
    loading = config["dense_diagonal_loading"]
    dense_diagonal_on = config["dense_diagonal_on"]
    diag_mask, correction = mask_diagonal(hic, cutoff, loading, ndiag_bins, dense_diagonal_on)
    
    if adj:
        vbead = config["beadvol"]
        vcell = config["grid_size"]**3
        diag_mask *= vbead/vcell
        
    if getcorrect:
        return diag_mask, correction
    else:
        return diag_mask
    

def mask_diagonal2(contact, bins=16):
    """Returns weighted averages of contact map"""
    rows, cols = contact.shape
    binsize = rows/bins
    
    assert(rows==cols), "contact map must be square"
    nbeads = rows
    measure = []
    correction = []

    for b in range(bins):            
        mask = np.zeros_like(contact)
        for r in range(rows):
            for c in range(cols):
                if int((r-c)/binsize) == b:
                    mask[r,c] = 1
                    mask[c,r] = 1

                
        measure.append(np.mean((mask*contact).flatten()))
        correction.append(np.sum(mask))

    measure = np.array(measure)
    correction = np.array(correction)
    return measure, correction

def get_seq_kmeans(hic, k):
    meanDist = kmeans.genomic_distance_statistics(hic)
    y_diag = kmeans.diagonal_preprocessing(hic, meanDist)
    seqs, labels = kmeans.get_k_means_seq(y_diag, k)
    seqs = seqs.T
    return seqs

def bin_chipseq(df, resolution, method="max"):
    step = resolution
    areas = []
    for i in range(int(min(df['start'])), int(max(df['start'])), step):
        sliced = df[(df['start']>i) & (df['start'] < i+step)]
        if method == "area":
            areas.append(np.trapz(sliced['value'], x=sliced['start']))
        if method == "max":
            areas.append(sliced['value'].max())
        if method == "mean":
            areas.append(sliced['value'].mean())
    return areas

def maxent_setup(dirname, k, hic_full, seqs_full=None, start=None, size=None, plot=True, nchis_diag=32, adjust=True, ncells=0):
    print("nchis_diag:", nchis_diag)
    seqs_full = np.array(seqs_full)
    print(np.shape(seqs_full))
    hic = hic_full
    if seqs_full is None:
        print("getting sequences")
        seqs = ep.get_sequences(hic, k, dtype="float", map11=True, map01=False, split=False)
        hic = hic_full
    
    if start is not None and size is not None:
        print("setting endpoints")
        end = start + size
        seqs = seqs_full[0:k, start:end]
        hic = hic_full[start:end, start:end]
        print("slicing")
        print("start: ", start)
        print("end: ", end)
    else:
        print(np.shape(seqs_full))
        seqs = seqs_full[0:k]
    
    if plot:
        print(np.shape(seqs))
        print(np.shape(hic))
        plot_contactmap(hic)
        plt.figure()
        plt.plot(seqs[0])
    
    defaults = "~/Documents/TICG-chromatin/maxent/defaults"

    nchis = int(k*(k+1)/2) # plaid chis
    nbeads = len(seqs[0])

    goals_plaid = get_goal_plaid(hic, seqs, k)
    goals_diag = get_goal_diag(hic, nchis_diag)
    
    if adjust:
        goals_plaid = goals_plaid * nbeads**2 # change_goals(goals_plaid, nbeads, ncells)
        goals_diag = goals_diag * nbeads**2 #change_goals(goals_diag, nbeads, ncells)

    # set up new directory
    newdir = dirname
    newdir_res = osp.join(newdir, "resources")
    os.system(" ".join(("mkdir", newdir)))
    os.system(" ".join(("cp -r", defaults, newdir_res)))

    np.savetxt(osp.join(newdir_res,"chis.txt"), np.zeros((2,nchis)), fmt="%.8f")
    np.savetxt(osp.join(newdir_res,"chis_diag.txt"), np.zeros((2,nchis_diag)), fmt="%.8f")
    np.savetxt(osp.join(newdir_res,"obj_goal.txt"), goals_plaid, newline=" ", fmt="%.8f")
    np.savetxt(osp.join(newdir_res,"obj_goal_diag.txt"), goals_diag, newline=" ", fmt="%.8f")

    chipseq_files = []
    for i in range(k):
        pcfname = "pcf"+str(i+1)+".txt"
        chipseq_files.append(pcfname)
        np.savetxt(osp.join(newdir_res, pcfname), seqs[i], newline="\n", fmt="%.8f")

    with open(osp.join(newdir_res, "config.json")) as f:
        config = json.load(f)

    with open(osp.join(newdir_res, "config.json"), "w") as f:
        config["nspecies"] = k
        config["bead_type_files"] = chipseq_files
        config["load_bead_types"] = True
        config["nbeads"] = nbeads
        config["load_configuration"] = False
        
        LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(k):
            for j in range(k):
                if j < i:
                    continue
                key = 'chi{}{}'.format(LETTERS[i], LETTERS[j])
                config[key] = 0  
                
        json.dump(config, f, indent=4)
    
def plot_scatter(hic1, hic2, label1="first", label2="second"):    
    plt.figure(figsize=(12,10))
    plt.plot(np.log10(hic1).flatten(), np.log10(hic2).flatten(), 'kx', markersize=0.1)
    plt.plot(np.log10(np.linspace(0.001,1,10)), np.log10(np.linspace(0.001,1,10)), 'r--')
    
    scc = get_SCC(hic1, hic2)
    pearson = get_pearson(hic1, hic2)
    rmse = get_RMSE(hic1, hic2)

    plt.title("Scatter (raw) Pearson: {%.2f}, RMSE: {%.2f}" %(pearson, rmse))

    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.axis([-4,0.5,-5, 0.5])
    
def plot_scatter_oe(hic1, hic2):    
    scc = get_SCC(hic1, hic2)
    pearson = get_pearson(hic1, hic2)
    rmse = get_RMSE(hic1, hic2)
    
    hic1 = get_oe(hic1)
    hic2 = get_oe(hic2)
    
    plt.figure(figsize=(12,10))
    plt.plot(np.log10(hic1).flatten(), np.log10(hic2).flatten(), 'kx', markersize=0.1)
    plt.plot(np.log10(np.linspace(0.001,100,10)), np.log10(np.linspace(0.001,100,10)), 'r--')
    
    plt.title("Scatter (normalized) SCC: {%.2f}" %(scc))
    plt.xlabel("log(OE_sim)")
    plt.ylabel("log(OE_exp)")
    plt.axis([-4,0.5,-5, 0.5])
    plt.axis([-2,2,-2, 2])
     
def get_SCC(hic1, hic2):
    #oe1 = get_oe(hic1)
    #oe2 = get_oe(hic2)
    #return np.corrcoef(oe1.flatten(), oe2.flatten())[0,1]
    myscc = SCC()
    return myscc.scc(hic1, hic2)

def get_RMSE(hic1, hic2):
    return np.sqrt(sklearn.metrics.mean_squared_error(hic1, hic2))
              
def get_RMSLE(hic1, hic2):
    return np.sqrt(sklearn.metrics.mean_squared_log_error(hic1, hic2))

def get_pearson(hic1, hic2):
    return np.corrcoef(hic1.flatten(), hic2.flatten())[0,1]

def fill_subdiagonal(a, offset, fn):
    n = np.shape(a)[0]
    inds = np.arange(n-offset)
    orig_subdiag = np.diag(a, offset)
    new_subdiag = fn(orig_subdiag)
    a[inds, inds+offset] = new_subdiag
    a[inds+offset, inds] = new_subdiag
    return a

def change_goals(goals, nbeads, ncells, vbead):
    return goals * nbeads**2 / (2 * vbead * 2 * ncells)

def get_goals(hic, seqs, config, diag_bins=32, save_path=None):
    plaid = get_goal_plaid(hic, seqs, config)
    diag = get_goal_diag(hic, config, diag_bins)
    
    if save_path is not None:
        np.savetxt(save_path, plaid, newline=" ", fmt="%.8f")
        np.savetxt(save_path, diag, newline=" ", fmt="%.8f")
        
    return np.hstack((plaid, diag))    

@njit
def reduce(hic, newrows, newcols):
    rows, cols = hic.shape

    sfx = int(rows/newrows) # shrink factor
    sfy = int(cols/newcols) # shrink factor

    newhic = np.zeros((newrows, newcols))

    for r in range(newrows):
        for c in range(newcols):
            x = r*sfx
            y = c*sfy
            newhic[r,c] = np.sum(hic[ x:x+sfx, y:y+sfy ])
    return newhic


@njit
def reduce_mean(hic, newrows, newcols):
    rows, cols = hic.shape

    sfx = int(rows/newrows) # shrink factor
    sfy = int(cols/newcols) # shrink factor

    newhic = np.zeros((newrows, newcols))

    for r in range(newrows):
        for c in range(newcols):
            x = r*sfx
            y = c*sfy
            newhic[r,c] = np.mean(hic[ x:x+sfx, y:y+sfy ])
    return newhic

@njit
def resize_contactmap(hic, newrows, newcols=None):
    rows, cols = hic.shape
    
    if newcols is None:
        newrows = newcols
        
    newhic = np.zeros((newrows, newcols))
        
    # increase the size of the contactmap
    if newrows > rows and newcols > cols:
        sfx = int(newrows/rows) # scale factor
        sfy = int(newcols/rows) # scale factor
        
        for x in range(rows):
            for y in range(cols):
                r = x*sfx
                c = y*sfy
                newhic[r:r+sfx, c:c+sfy] = hic[x,y] / np.sqrt(sfx)
        
        diag = np.zeros(newrows)
        for i in range(newrows):
            diag[i] = newhic[i, i]
            
        #newhic /= np.mean(diag)
        newhic /= np.max(newhic)
    
    # decrease the size of the contactmap
    if newrows < rows and newcols < cols:
        sfx = int(rows/newrows) # shrink factor
        sfy = int(cols/newcols) # shrink factor
        
        for r in range(newrows):
            for c in range(newcols):
                x = r*sfx
                y = c*sfy
                newhic[r,c] = np.sqrt(sfx)*np.mean(hic[ x:x+sfx, y:y+sfy ])
      
        diag = np.zeros(newrows)
        for i in range(newrows):
            diag[i] = newhic[i, i]
            
        #newhic /= np.mean(diag)        
        #diag = np.zeros(rows)
        #for i in range(rows):
            #diag[i] = newhic[i, i]        
        
        #newhic /= np.mean(diag)
        
        # I think dividing by the max is the exact correct method
        # but outliers in the experimental data are screwing things up, 
        # so try dividing by the diagonal (above)
        newhic /= np.max(newhic)
        
    # no change to size of contact map          
    if newrows == rows and newcols == cols:
        newhic = hic
 
    return newhic

@njit
def reduce_sqrt(hic, newrows, newcols):
    rows, cols = hic.shape

    sfx = int(rows/newrows) # shrink factor
    sfy = int(cols/newcols) # shrink factor

    newhic = np.zeros((newrows, newcols))

    for r in range(newrows):
        for c in range(newcols):
            x = r*sfx
            y = c*sfy
            newhic[r,c] = np.sqrt(sfx)*np.mean(hic[ x:x+sfx, y:y+sfy ])
    return newhic

@njit
def amplify_sqrt(hic, newrows, newcols):
    rows, cols = hic.shape

    sfx = int(newrows/rows) # scale factor
    sfy = int(newcols/rows) # scale factor

    newhic = np.zeros((newrows, newcols))

    for x in range(rows):
        for y in range(cols):
            r = x*sfx
            c = y*sfy
            newhic[r:r+sfx, c:c+sfy] = hic[x,y] / np.sqrt(sfx)
            
    return newhic
   

@njit
def reduce2(hic, newrows, newcols, fn=np.sum):
    rows, cols = hic.shape

    sfx = int(rows/newrows) # shrink factor
    sfy = int(cols/newcols) # shrink factor

    newhic = np.zeros((newrows, newcols))

    for r in range(newrows):
        for c in range(newcols):
            x = r*sfx
            y = c*sfy
            newhic[r,c] = fn(hic[ x:x+sfx, y:y+sfy ])
    return newhic

def vacancy(hic, plot=True):
    '''what percentage of the input image pixels are zero?'''
    
    sz = np.shape(hic)[0]
    x = np.zeros_like(hic)
    x[np.where(hic==0)] = 1

    vacancy = np.sum(x)/sz**2
    
    if plot:
        plt.figure(figsize=(12,10))
        plt.imshow(x, cmap='binary', vmin = 0, vmax= 1)
        plt.colorbar()
        plt.title("Vacancy: {:.1f} %".format(vacancy*100))
    return vacancy

def plot_consistency(sim):
        
    # can only calculate consistency if the hic resolution is the same
    # as the observables resolution
    if (sim.config['contact_resolution'] == 1):
        if np.shape(sim.hic) != np.shape(sim.seqs):
            size = np.shape(sim.seqs[0])[0]
            hic = resize_contactmap(sim.hic, size, size)
        else:
            hic = sim.hic

        plaid = get_goal_plaid(hic, sim.seqs, sim.config)
        diag = get_goal_diag(hic, sim.config, sim.diag_bins, dense_diagonal_on=sim.config["dense_diagonal_on"])
        goal = np.hstack((plaid,diag))


        diff = sim.obs_tot - goal
        error = np.sqrt(diff@diff / (goal@goal))

        plt.figure()
        plt.plot(sim.obs_tot, 'o', label="obs")
        plt.plot(goal, 'x', label="goal")
        plt.title("sim.obs vs Goals(sim.hic); Error: {%.3f}"%error)
        plt.legend()
        return error
        
    elif sim.config['contact_resolution'] > 1:
        print("cannot plot consistency because sim.config['contact_resolution'] >1")
        return 0

def get_symmetry_score(A, order='fro'):
    symmetric = np.linalg.norm(1/2*(A + A.T), order)
    skew_symmetric = np.linalg.norm(1/2*(A - A.T), order)
    
    return symmetric / (symmetric + skew_symmetric)    

def get_symmetry_score_2(first, second, order='fro'):
    composite = make_tri_composite(first, second)
    return get_symmetry_score(composite, order)

def make_tri_composite(first, second):
    npixels = np.shape(first)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)
    
    composite = np.zeros((npixels, npixels))
    
    composite[indu] = first[indu]
    composite[indl] = second[indl]
    return composite
    
def plot_tri(first, second, vmaxp=None, oe=False, title="", dark=False, log=False):
    
    if dark:
        vmaxp=np.mean(first)/2
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
        plot_fn(composite, vmaxp=vmaxp, absolute=True, title=title, log=log)

    plt.title("symmetry score: {%.2f}"%(symmetry_score))

def plot_obs_vs_goal(sim):
    fig, axs = plt.subplots(2, figsize=(12,14))
    obs = sim.obs_tot
    goal = sim.obj_goal_tot
    axs[0].plot(obs, '--o', label="obs")
    axs[0].plot(goal, 'ko', label="goal")
    axs[0].vlines(len(sim.obs), min(sim.obs_tot)/5, max(sim.obs_tot)/5, 'k')
    axs[0].legend()
    axs[0].set_title("Observables vs Goals")

    diff = sim.obs_tot - sim.obj_goal_tot
    axs[1].plot(diff, '--o')
    axs[1].hlines(0, len(sim.obs_tot), 0, 'k')
    axs[1].vlines(len(sim.obs), min(diff)/5, max(diff)/5, 'k')
    axs[1].set_title("Difference")
    
    
class parameters():
    def __init__(self, N):
        self.N = N
        self.baseN = 512000
        self.basev = 520
        self.baseb = 16.5
        self.side_length = 1643 # nm
        self.frac = self.baseN/self.N
        
        self.v = self.basev * self.frac
        self.b = self.baseb * np.sqrt(self.frac)
        
    
    def convertv(self, N):
        return self.basev * self.frac
    
    def convertb(self, N):
        return self.baseb * np.sqrt(self.frac)
    
    def delta(self, nint):
        self.nsite = self.N/nint
        self.L = self.nsite**(1/3)
        return self.side_length/self.L
    
    def nint(self, delta):
        L = self.side_length / delta
        nsites = L**3
        nint = self.N / nsites
        return nint

def load_contactmap_hicstraw(hicfile, res, chrom, start, end, clean=False, KR=True):
    assert(res in hicfile.getResolutions())
    
    if KR:
        mzd = hicfile.getMatrixZoomData(chrom, chrom, "observed", "KR", "BP", res)
    else:
        mzd = hicfile.getMatrixZoomData(chrom, chrom, "observed", "NONE", "BP", res)
    
    if res > 50000:
        contact = mzd.getRecordsAsMatrix(start,end,start,end)
    else:
        contact = load_contactmap_with_buffers(mzd, start, end, res)

    if clean:
        contact, dropped_indices = clean_contactmap(contact)

    return contact

def initialize(hicfile, res, size, randomized=False, chrom='2'):
    start = 0
    end = 120_000_000

    contact = load_contact_hicstraw(hicfile, res, chrom)
    print("loaded contactmap, shape: ", np.shape(contact))
    
    clean_contact, inds = clean_contactmap(contact)
    print("cleaned contactmap, shape: ", np.shape(clean_contact))
    
    seqs = get_sequences(clean_contact, 5, dtype="float", randomized=randomized)
    
    contact_out = get_contactmap(clean_contact, normtype="mean") # normalizes
    
    return contact_out[0:size,0:size], seqs[:,0:size], inds 
    #return contact_out, seqs, inds
        

def load_contactmap_with_buffers(mzd, start, end, res):
    bufsize = 5_000_000 #increment
    width = end-start
    
    print(width)
    assert(width % res == 0)
    assert(width % bufsize == 0)
    
    pixels = int(width/res)
    steps = int(width/bufsize)
    
    assert(pixels % steps == 0)
    pix_per_buf = int(pixels/steps)
    print(pixels)
    
    out = np.zeros((pixels, pixels))
    
    for i in tqdm(range(steps)):
        for j in range(steps):
            xmin = i*bufsize
            xmax = (i+1)*bufsize
            ymin = j*bufsize
            ymax = (j+1)*bufsize
            
            imin = i*pix_per_buf
            imax = (i+1)*pix_per_buf
            jmin = j*pix_per_buf
            jmax = (j+1)*pix_per_buf
            
            buffer = mzd.getRecordsAsMatrix(xmin, xmax-1, ymin, ymax-1)
            out[imin:imax, jmin:jmax] = buffer
            
    return out

def make_clean_mask(inds, N):
    mask = np.full((N,N), True)
    for i in inds:
        mask[i, :] = False
        mask[:, i] = False
        
    return mask
    

def clean_contactmap(contact):
    N, _  = np.shape(contact)
    d  = np.diagonal(contact)
    inds = np.where(d == 0)[0] 
    mask = make_clean_mask(inds, N)
    deleted  = len(inds)
                
    return contact[mask].reshape(N-deleted, N-deleted), inds

def scale_sim(orig, factor, overwrite=False):
    '''
    orig: ep.Sim object of original simulation
    factor: factor by which to scale to new simulation
    '''
    newsim = copy.deepcopy(orig)
    #newbeads = 32768*2*2*2*2
    newbeads = orig.config['nbeads']*factor
    newsim.config['nbeads'] = newbeads
    #factor = newbeads/orig.config['nbeads']
    newsim.config['bond_length'] = orig.config['bond_length'] / np.sqrt(factor)
    newsim.config['beadvol'] = orig.config['beadvol'] / factor
    newsim.config['diag_chis'] = list(np.array(orig.config['diag_chis']) / factor)
    newsim.config['load_configuration'] = False

    newsim.chi /= factor
    newsim.save_chis()

    k, n = newsim.seqs.shape
    factor = newbeads/orig.config['nbeads']
    # assert facttor is an integer!
    factor = int(factor)
    newseqs = np.zeros((k, newsim.config['nbeads']))
    for s, seq in enumerate(newsim.seqs):
        for e, ele in enumerate(seq):
            for i in range(factor):
                newseqs[s, factor*e+i] = seq[e]

    newsim.seqs = newseqs
    newsim.config['nSweeps'] = 100000

    newsim.path = str(newbeads)

    newsim.init_sim(overwrite = overwrite, getgoals = False)

def rescale_matrix(inp, factor, method="sum"):
    '''
    Rescales input matrix by factor.
    if inp is 1024x1024 and factor=2, out is 512x512
    '''

    assert len(inp.shape) == 2, f'must be 2d array not {inp.shape}'
    m, _ = inp.shape
    assert m % factor == 0, f'factor must evenly divide m {m}%{factor}={m%factor}'
    inp = np.triu(inp) # need triu to not double count entries

    if method=="sum":
        fn = np.sum
    if method=="mean":
        fn = np.mean

    processed = skimage.measure.block_reduce(inp, (factor, factor), fn)
    # need to make symmetric again
    processed = np.triu(processed)
    out = processed + np.triu(processed, 1).T 
    
    return out
