import math
import multiprocessing
import os
import os.path as osp

import hicrep
import numpy as np
import pylib.utils.hic_utils as hic_utils
import scipy
import sklearn.metrics
from pylib.utils import epilib
from pylib.utils.utils import triu_to_full
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr, zscore


# similarity measures
class SCC():
    """
    Calculate Stratified Correlation Coefficient (SCC)
    as defined by https://pubmed.ncbi.nlm.nih.gov/28855260/.
    """
    def __init__(self, h=1, K=None, var_stabilized=True, start=None):
        self.r_2k_dict = {} # memoized solution for var_stabilized r_2k
        self.h = h
        self.K = K
        if start is None:
            self.start = 0
        else:
            self.start = start
        self.var_stabilized = var_stabilized


    def r_2k(self, x_k, y_k):
        '''
        Compute r_2k (numerator of pearson correlation)

        Inputs:
            x: contact map
            y: contact map of same shape as x
            var_stabilized: True to use var_stabilized version
        '''

        # x and y are stratums
        if self.var_stabilized:
            # var_stabilized computes variance of ranks
            N_k = len(x_k)
            if N_k in self.r_2k_dict:
                result = self.r_2k_dict[N_k]
            else:
                # variance is permutation invariant, so just use np.arange instead of computing ranks
                # futher var(rank(x_k)) = var(rank(y_k)), so no need to compute for each
                # this allows us to memoize the solution via self.r_2k_dict (solution depends only on N_k)
                # memoization offers a marginal speedup when computing lots of scc's
                result = np.var(np.arange(1, N_k+1)/N_k, ddof = 1)
                self.r_2k_dict[N_k] = result
            return result
        else:
            return math.sqrt(np.var(x_k) * np.var(y_k))

    def scc_file(self, xfile, yfile, h=None, K=None,
                verbose=False, distance=False, chr=None, resolution=None):
        '''
        Wrapper for scc that takes file path as input. Must be .npy file.

        Inputs:
            xfile: file path to contact map
            yfile: file path to contact map
            ...: see self.scc()
            chr: chromosome if mcool files
            resolution: resoultion if mcool files
        '''
        if xfile == yfile:
            # no need to compute
            result = 1 - distance
            if verbose:
                return result, None, None
            else:
                return result

        x = hic_utils.load_contact_map(xfile, chr, resolution)
        y = hic_utils.load_contact_map(yfile, chr, resolution)
        return self.scc(x, y, h, K, verbose, distance)

    def mean_filter(self, x, size):
        return scipy.ndimage.uniform_filter(x, size, mode="constant") / (size ** 2)

    def scc(self, x, y, K=None, verbose=False,
            debug=False, distance=False):
        '''
        Compute scc between contact map x and y.

        Inputs:
            x: contact map
            y: contact map of same shape as x
            h: span of mean filter (width = (1+2h)) (None or 0 to skip)
            K: maximum stratum (diagonal) to consider (5 Mb recommended)
            verbose: True to print when nan found
            debug: True to return p_arr and w_arr
            distance: True to return 1 - scc
        '''
        if len(x.shape) == 1:
            x = triu_to_full(x)
        if len(y.shape) == 1:
            y = triu_to_full(y)

        if K is None:
            if self.K is None:
                K = len(y) - 2
            else:
                K = self.K


        if self.h is not None:
            x = self.mean_filter(x.astype(np.float64), 1+2*self.h)
            y = self.mean_filter(y.astype(np.float64), 1+2*self.h)

        nan_list = []
        p_arr = []
        w_arr = []
        for k in range(self.start, K):
            # get stratum (diagonal) of contact map
            x_k = np.diagonal(x, k)
            y_k = np.diagonal(y, k)


            # filter to subset of diagonals where at least 1 is nonzero
            # i.e if x_k[i] == y_k[i] == 0, ignore element i
            # use 1e-12 for numerical stability
            x_zeros = np.argwhere(abs(x_k)<1e-12)
            y_zeros = np.argwhere(abs(y_k)<1e-12)
            both_zeros = np.intersect1d(x_zeros, y_zeros)
            mask = np.ones(len(x_k), bool)
            mask[both_zeros] = 0
            x_k = x_k[mask]
            y_k = y_k[mask]

            N_k = len(x_k)

            if N_k > 1:
                p_k, _ = pearsonr(x_k, y_k)

                if np.isnan(p_k):
                    # ignore nan
                    nan_list.append(k)
                else:
                    p_arr.append(p_k)
                    r_2k = self.r_2k(x_k, y_k)
                    w_k = N_k * r_2k
                    w_arr.append(w_k)

        w_arr = np.array(w_arr)
        p_arr = np.array(p_arr)

        scc = np.sum(w_arr * p_arr / np.sum(w_arr))

        if verbose and len(nan_list) > 0:
            print(f'{len(nan_list)} nans: k = {nan_list}')

        if distance:
            scc =  1 - scc

        if debug:
            return scc, p_arr, w_arr
        else:
            return scc


def hicrep_scc(fmcool1, fmcool2, h, K, binSize=-1, distance = False):
    '''
    Compute scc between contact map x and y.

    Inputs:
        x: contact map
        y: contact map of same shape as x
        h: span of mean filter (width = (1+2h)) (None to skip)
        K: maximum stratum (diagonal) to consider (None for all) (5 Mb recommended)
        distance: True to return 1 - scc
    '''
    cool1, binSize1 = readMcool(fmcool1, binSize)
    cool2, binSize2 = readMcool(fmcool2, binSize)
    if binSize == -1:
        assert binSize1 == binSize2, f"bin size mismatch: {binSize1} vs {binSize2}"
        binSize = binSize1


    scc = hicrep.hicrepSCC()
    if distance:
        return 1 - scc
    else:
        return scc


class InnerProduct():
    '''Based off of InnerProduct from https://github.com/liu-bioinfo-lab/scHiCTools'''
    def __init__(self, dir = None, files = None, K = 10, jobs = 10,
                resolution = None, chr = None):
        '''
        Inputs:
            dir: directory containing input data (required if files is None)
            files: list of input file paths (required if dir is None)
            K: maximum stratum to consider
            jobs: number of jobs for multiprocessing
            resolution: resolution of input data (required if .mcool files)
            chr: chromosome of input data (required if .mcool files)
        '''
        self.K = K
        self.resolution = resolution
        self.chr = chr
        if dir is not None:
            self.files = [osp.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]
        else:
            assert files is not None
            self.files = files


        if jobs > 1:
            with multiprocessing.Pool(jobs) as p:
                self.zscores = p.map(self.get_zscore_feature, self.files)
        else:
            self.zscores = []
            for f in files:
                self.zscores.append(self.get_zscore_feature(f))
        self.zscores = np.array(self.zscores)

    def get_distance_matrix(self):
        inner = self.zscores.dot(self.zscores.T) / self.zscores.shape[1]
        inner[inner > 1] = 1
        inner[inner < -1] = -1
        distance_mat = np.sqrt(2 - 2 * inner)

        return distance_mat

    def get_zscore_feature(self, file):
        y = hic_utils.load_contact_map(file, self.chr, self.resolution)
        if len(y.shape) == 1:
            y = triu_to_full(y)
        zscores = []
        for k in range(self.K):
            y_k = np.diagonal(y, k)
            z = zscore(y_k)
            z[np.isnan(z)] = 0
            zscores.append(z)
        return np.concatenate(zscores)


def get_symmetry_score(A, order="fro"):
    symmetric = np.linalg.norm(1 / 2 * (A + A.T), order)
    skew_symmetric = np.linalg.norm(1 / 2 * (A - A.T), order)

    return symmetric / (symmetric + skew_symmetric)

def hic_spector(y1, y2, num_evec):
    '''HiC-spector metric from https://github.com/gersteinlab/HiC-spector.'''
    def evec_distance(v1,v2):
        d1=np.dot(v1-v2,v1-v2)
        d2=np.dot(v1+v2,v1+v2)
        if d1<d2:
            d=d1
        else:
            d=d2
        return np.sqrt(d)

    def get_ipr(evec):
        ipr=1.0/(evec*evec*evec*evec).sum()
        return ipr

    def get_Laplacian(M):
        S=M.sum(1)
        i_nz=np.where(S>0)[0]
        S=S[i_nz]
        M=(M[i_nz].T)[i_nz].T
        S=1/np.sqrt(S)
        M=S*M
        M=(S*M.T).T
        n=np.size(S)
        M=np.identity(n)-M
        M=(M+M.T)/2
        return M

    M1=lil_matrix(y1)
    M2=lil_matrix(y2)

    k1=np.sign(M1.A).sum(1)
    d1=np.diag(M1.A)
    kd1=~((k1==1)*(d1>0))
    k2=np.sign(M2.A).sum(1)
    d2=np.diag(M2.A)
    kd2=~((k2==1)*(d2>0))
    iz=np.nonzero((k1+k2>0)*(kd1>0)*(kd2>0))[0]
    M1b=(M1[iz].A.T)[iz].T
    M2b=(M2[iz].A.T)[iz].T

    i_nz1=np.where(M1b.sum(1)>0)[0]
    i_nz2=np.where(M2b.sum(1)>0)[0]
    i_z1=np.where(M1b.sum(1)==0)[0]
    i_z2=np.where(M2b.sum(1)==0)[0]

    M1b_L=get_Laplacian(M1b)
    M2b_L=get_Laplacian(M2b)

    a1, b1=eigsh(M1b_L,k=num_evec,which="SM")
    a2, b2=eigsh(M2b_L,k=num_evec,which="SM")

    b1_extend=np.zeros((np.size(M1b,0),num_evec))
    b2_extend=np.zeros((np.size(M2b,0),num_evec))
    for i in range(num_evec):
        b1_extend[i_nz1,i]=b1[:,i]
        b2_extend[i_nz2,i]=b2[:,i]

    ipr_cut=5
    ipr1=np.zeros(num_evec)
    ipr2=np.zeros(num_evec)
    for i in range(num_evec):
        ipr1[i]=get_ipr(b1_extend[:,i])
        ipr2[i]=get_ipr(b2_extend[:,i])

    b1_extend_eff=b1_extend[:,ipr1>ipr_cut]
    b2_extend_eff=b2_extend[:,ipr2>ipr_cut]
    num_evec_eff=min(np.size(b1_extend_eff,1),np.size(b2_extend_eff,1))

    evd=np.zeros(num_evec_eff)
    for i in range(num_evec_eff):
        evd[i]=evec_distance(b1_extend_eff[:,i],b2_extend_eff[:,i])

    Sd=evd.sum()
    l=np.sqrt(2)
    evs=abs(l-Sd/num_evec_eff)/l

    N=float(M1.shape[1]);
    if (np.sum(ipr1>N/100)<=1)|(np.sum(ipr2>N/100)<=1):
        print("at least one of the maps does not look like typical Hi-C maps")
    # else:
    #     print("size of maps: %d" %(np.size(M1,0)))
    #     print("reproducibility score: %6.3f " %(evs))
    #     print("num_evec_eff: %d" %(num_evec_eff))
    return evs

def genome_disco(y1, y2, t):
    def normalize(y):
        # normalize y such that rows sum to 1
        row_sums = y.sum(axis=1, keepdims=True)
        y /= row_sums

    normalize(y1)
    normalize(y2)

    norm = np.sum(np.abs(np.power(y1, t) - np.power(y2, t)))

    denom = 0.5 * (np.sum(np.sum(y1, axis=1) > 0) +  np.sum(np.sum(y2, axis=1) > 0))

    diff_score = norm / denom

    score = 1 - diff_score

    return score

## wrapper functions - Soren ##
def get_SCC(hic1, hic2):
    # oe1 = get_oe(hic1)
    # oe2 = get_oe(hic2)
    # return np.corrcoef(oe1.flatten(), oe2.flatten())[0,1]
    return SCC().scc(hic1, hic2)

def get_RMSE(hic1, hic2):
    return np.sqrt(sklearn.metrics.mean_squared_error(hic1, hic2))

def get_RMSLE(hic1, hic2):
    return np.sqrt(sklearn.metrics.mean_squared_log_error(hic1, hic2))
## ##

def get_pearson(hic1, hic2):
    return np.corrcoef(hic1.flatten(), hic2.flatten())[0, 1]

def test():
    y1 = np.load('/home/erschultz/dataset_12_06_23/samples/sample1/y.npy')
    # y2 = np.load('/home/erschultz/dataset_12_06_23/samples/sample2/y.npy')
    # y1 = np.load('/home/erschultz/dataset_11_20_23/samples/sample3/y.npy')
    # y2 = np.load('/home/erschultz/dataset_11_20_23/samples/sample202/y.npy')
    y2 = np.load('/home/erschultz/dataset_12_06_23/samples/sample1/optimize_grid_b_200_v_8_spheroid_1.5-max_ent10/iteration30/y.npy')
    y1 = np.eye(10)
    y2 = np.ones((10, 10))

    # s = genome_disco(epilib.get_oe(y1), epilib.get_oe(y2), 3)
    # s = genome_disco(y1, y2, 3)
    # print('score', s)

    # files = ['/home/erschultz/scratch/contact_diffusion_kNN17inner_product/iteration_0/sc_contacts/y_sc_76.npy',
    #         '/home/erschultz/scratch/contact_diffusion_kNN17inner_product/iteration_0/sc_contacts/y_sc_78.npy']
    # IP = InnerProduct(files = files, K = 20, jobs = 1,
    #                 resolution = None, chr = None)
    # D = IP.get_distance_matrix()
    # print(D)

    scc = SCC()
    scc, p_arr, w_arr = scc.scc(y1, y2, debug = True, var_stabilized=True)
    print(w_arr, w_arr.shape)


if __name__ == '__main__':
    test()
