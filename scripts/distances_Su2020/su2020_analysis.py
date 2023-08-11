import csv
import json
import math
import os
import os.path as osp
import string
import sys

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import seaborn as sns
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import (BLUE_CMAP, BLUE_RED_CMAP,
                                        RED_BLUE_CMAP, RED_CMAP, plot_matrix,
                                        plot_mean_dist, rotate_bound)
from pylib.utils.similarity_measures import SCC
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import ArgparserConverter
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_import_log, load_Y)
from sequences_to_contact_maps.scripts.utils import (calc_dist_strat_corr,
                                                     nan_pearsonr,
                                                     pearson_round)
from sequences_to_contact_maps.scripts.xyz_utils import (calculate_rg,
                                                         xyz_load,
                                                         xyz_to_distance,
                                                         xyz_write)


# plotting functions
def plot_diagonal(exp, sim, ofile=None):
    npixels = np.shape(sim)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)

    # make composite contact map
    composite = np.zeros((npixels, npixels))
    composite[indu] = sim[indu]
    composite[indl] = exp[indl]
    print(composite, composite.shape)

    # Rotate 45 degs
    resized = rotate_bound(composite,-45)
    print(resized, resized.shape)
    vmax = np.nanmean(composite)

    # crop
    center = resized.shape[0] // 2
    height=30
    resized = resized[center-height:center+height, :]


    fig, ax = plt.subplots(figsize=(6, 3),dpi=600)
    # _pf = ax.imshow(resized, vmin=200, vmax=1000, cmap='seismic')
    sns.heatmap(resized, ax = ax, linewidth = 0, vmin = 0, vmax = vmax, cmap = BLUE_CMAP)
    ax.set_yticks([])

    if ofile is not None:
        plt.savefig(ofile)
        plt.close()
    else:
        plt.show()

def plot_distance_map(input, dir, label):
    plot_matrix(input, osp.join(dir, f'D_{label}.png'), f'D_{label}', vmax = 'max', cmap=RED_BLUE_CMAP)
    # normalize by genomic distance
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(input)
    input_diag = DiagonalPreprocessing.process(input, meanDist)

    plot_matrix(input_diag, osp.join(dir, f'D_diag_{label}.png'), f'D_diag_{label}', vmin = 'center1', cmap='bluered')

    if np.sum(np.isnan(input_diag)) == 0:
        input_diag_corr = np.corrcoef(input_diag)
        plot_matrix(input_diag_corr, osp.join(dir, f'D_diag_corr_{label}.png'), f'D_diag_corr_{label}', vmin = -1, vmax= 1, cmap='bluered')

# utility functions
def dist_distribution_a_b(xyz, a, b):
    # a and b are indices
    num_cells, num_coords, _ = xyz.shape
    dist = np.zeros(num_cells)
    for i in range(num_cells):
        dist[i] = np.linalg.norm(xyz[i, a, :] - xyz[i, b, :])
    return dist

def dist_distribution_seq(D, seq_a, seq_b):
    # seq_a and seq_b are binary vectors
    num_cells, num_coords, _ = D.shape
    num_cells = min(100, num_cells)
    dist = []
    mask_aa = np.outer(seq_a, seq_a)
    mask_bb = np.outer(seq_b, seq_b)
    mask_ab = np.outer(seq_a, seq_b)

    all_aa = np.zeros(num_cells*np.sum(mask_aa))
    all_bb = np.zeros(num_cells*np.sum(mask_bb))
    all_ab = np.zeros(num_cells*np.sum(mask_ab))
    for i in range(num_cells):
        D_i = D[i]
        aa = D_i[mask_aa]
        all_aa[len(aa)*i:len(aa)*i+len(aa)] = aa

        bb = D_i[mask_bb]
        all_bb[len(bb)*i:len(bb)*i+len(bb)] = bb

        ab = D_i[mask_ab]
        all_ab[len(ab)*i:len(ab)*i+len(ab)] = ab

    return all_aa, all_bb, all_ab

def crop_hg38(dir, inp, start, m):
    '''
    Crop input distance map to m pixels starting from start.

    Inputs:
        inp (np array): distance map
        start (str): genomic coordinates
        m (int): desired len of output
    '''
    coords_file = osp.join(dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)

    i = coords_dict[start]
    print(i)
    if len(inp.shape) == 2:
        return inp[i:i+m, i:i+m]
    elif len(inp.shape) == 3:
        return inp[:, i:i+m, i:i+m]
    else:
        raise Exception(f'Unaccepted shape: {inp.shape}')

def hg38_to_hg19(coords, chr=21):
    '''Convert hg38 coords to hg19.'''
    if chr == 21:
        dir = '/home/erschultz/Su2020/samples/sample1'
    with open(osp.join(dir, 'hg38_positions.txt'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            if line.strip() == coords:
                break

    with open(osp.join(dir, 'hg19_positions.bed')) as f:
        return f.readlines()[i]

def min_MSE(D, D_sim):
    def metric(alpha, D, D_sim):
        return mean_squared_error(D, D_sim*alpha)

    result = minimize(metric, x0=1, args=(D, D_sim))
    print(f'resultt {result}')


    return result.x

def rescale_mu_sigma(D, D_sim, return_params = False):
    mu_D_sim = np.nanmean(D_sim)
    mu_D = np.nanmean(D)
    sigma_D_sim = np.nanstd(D_sim)
    sigma_D = np.nanstd(D)

    if return_params:
        return mu_D_sim, sigma_D_sim, mu_D, sigma_D
    else:
        return (D_sim - mu_D_sim)/sigma_D_sim * sigma_D + mu_D

def get_pcs(input, nan_rows=None, verbose=False, smooth=False, h=1):
    if input is None:
        return None
    if nan_rows is None:
        nan_rows = np.isnan(input[0])
    input = input[~nan_rows][:, ~nan_rows] # ignore nan_rows

    if smooth:
        input = scipy.ndimage.gaussian_filter(input, (h, h))

    input = epilib.get_oe(input)
    input = np.corrcoef(input)

    if verbose:
        print(input)
    seqs = epilib.get_pcs(input, 12,
                        normalize = False, align = True)
    return seqs.T # k x m

def tsv_to_npy():
    dir = '/home/erschultz/Su2020/samples/sample10'
    df = pd.read_csv(osp.join(dir, 'Hi-C_contacts_chromosome2.tsv'),
                    sep = '\t', index_col=0)
    y = df.to_numpy()
    np.save(osp.join(dir, 'y.npy'), y)
    plot_matrix(y, osp.join(dir, 'y.png'), vmax='mean')
    y_512 = y[-512:, -512:]
    print(y_512.shape)

    odir = '/home/erschultz/Su2020/samples/sample11'
    os.mkdir(odir, mode=0o755)
    np.save(osp.join(odir, 'y.npy'), y_512)
    plot_matrix(y_512, osp.join(odir, 'y.png'), vmax='mean')

def to_proximity(D, cutoff=500):
    N, _, _ = D.shape
    where = D < cutoff
    D_cutoff = np.zeros_like(D)
    D_cutoff[where] = 1
    # D_cutoff is 1 if in contact, ele 0

    D_prox = np.sum(D_cutoff, axis=0)
    D_prox /= N
    # D_prox is p(contact)

    return D_prox


# analysis scripts
def find_hg38_positions():
    dir = '/home/erschultz/Su2020/samples/sample10'
    file = osp.join(dir, 'Hi-C_contacts_chromosome2.tsv') # hic file
    df = pd.read_csv(file, sep = '\t', index_col=0)
    m=1024
    lower_ind = 0
    upper_ind = lower_ind + m

    # df.columns contains genomic coordinate windows
    cols = [i.split(':')[1].split('-') for i in df.columns]

    cols = [(int(i), int(j)) for i,j in cols]
    print(cols) # cols now contains tuples of (start, end) coordinate for each window

    # find contiguous region of 512 windows
    for i, (start, end) in enumerate(cols):
        if i <= lower_ind:
            prev_end = end
            continue
        if i > upper_ind:
            continue

        if not start == prev_end:
            diff = start-prev_end
            print(f'{i} {prev_end} {start}, diff={diff}, {diff//50000}')
            upper_ind -= diff // 50000

        prev_end = end

    columns = df.columns[lower_ind:upper_ind+1]
    cols = cols[lower_ind:upper_ind+1]
    print(lower_ind, upper_ind)
    print(columns)

    print((cols[-1][1] - cols[0][0]) / 50000)

    with open(osp.join(dir, f'hg38_positions.txt'), 'w') as f:
        wr = csv.writer(f, delimiter='\n')
        wr.writerows([columns])

def find_volume():
    '''Find average volume of cells in dataset.'''
    dir = '/home/erschultz/Su2020/samples/sample1'
    xyz_file = osp.join(dir, 'xyz.npy')
    xyz = np.load(xyz_file)

    print(xyz.shape, xyz.dtype)
    num_cells, num_coords, _ = xyz.shape
    vols = np.zeros(num_cells)
    for i in range(num_cells):
        xyz_i = xyz[i]
        xyz_i = xyz_i[~np.isnan(xyz_i)].reshape(-1, 3)
        points = len(xyz_i)
        if points < 100:
            print(f'Insufficient points for cell {i}')
            vols[i] = np.NaN
        else:
            try:
                hull = ConvexHull(xyz_i)
            except Exception:
                print(xyz[i])
                print(i, xyz_i.shape)
                raise
            vols[i] = hull.volume * 1e-9 # convert to um^3

    plt.hist(vols, bins = 50)
    plt.xlabel(r'Volume $\mu m^3$')
    plt.show()
    mean_vol = np.nanmean(vols)
    median_vol = np.nanmedian(vols)
    std_vol = np.nanstd(vols)
    print(f'median vol {mean_vol} um^3')
    print(f'mean vol {median_vol} um^3')
    print(f'std vol {std_vol}')
    r_sphere = (3/4 * mean_vol / np.pi)**(1/3)
    r_sphere *= 1e3
    print(f'spherical radius {r_sphere} nm')

def compare_dist_distribution_a_b(sample):
    '''Compare distributions of A-B distances between sim and experiment.'''

    dir = '/home/erschultz/Su2020/samples/sample1'
    xyz_file = osp.join(dir, 'xyz.npy')
    xyz = np.load(xyz_file)
    coords_file = osp.join(dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)

    coords_a = 'chr21:29100001-29150001'
    # coords_b = 'chr21:29400001-29450001'
    coords_b = 'chr21:28800001-28850001'

    a = coords_dict[coords_a]
    b = coords_dict[coords_b]
    dist = dist_distribution_a_b(xyz, a, b)

    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID)
    final_dir = get_final_max_ent_folder(max_ent_dir)
    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_sim = xyz_load(file, multiple_timesteps = True)
    result = load_import_log(dir)
    start = result['start']
    print(start)
    coords_a = hg38_to_hg19(coords_a)
    coords_b = hg38_to_hg19(coords_b)
    def get_ind_hg19(start, coords, res=50000):
        chr, rest = coords.split(':')
        l, u = rest.split('-')
        l = int(l); u = int(u)
        i=0
        while start < l:
            start += res
            i += 1
        end = start + res
        return i, f'{chr}:{l}-{u}'
    a, temp = get_ind_hg19(start, coords_a)
    b, temp = get_ind_hg19(start, coords_b)

    dist_sim = dist_distribution_a_b(xyz_sim, a, b)


    print('exp', dist, dist.shape)
    print('sim', dist_sim, dist_sim.shape)

    bin_width = 50
    arr = dist[~np.isnan(dist)]
    plt.hist(arr, label = 'Experiment', alpha = 0.5, color = 'black',
                weights = np.ones_like(arr) / len(arr),
                bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width))
    arr = dist_sim
    plt.hist(arr, label = 'Max. Ent.', alpha = 0.5, color = 'blue',
                weights = np.ones_like(arr) / len(arr),
                bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width))
    plt.xlabel('Distance between A and B', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.legend(fontsize=16)
    plt.title(f'Distance between\n{coords_a} and {coords_b}')
    # plt.title(f'Distance between A and B')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'distance_distribution.png'))
    plt.close()

def xyz_to_dist():
    '''Convert experimental xyz to distance map.'''
    chr=21
    dir = '/home/erschultz/Su2020/samples/sample1'
    log_file = osp.join(dir, 'xyz_to_dist.log')
    log_file = open(log_file, 'w')
    file = osp.join(dir, f'chromosome{chr}.tsv')
    xyz_file = osp.join(dir, 'xyz.npy')
    coords_file = osp.join(dir, 'coords.json')

    # get size of chromosome
    with open(file, 'r') as f:
        line = f.readline()
        line = f.readline().strip()
        line = line.split('\t')
        print(line)
        coords = line[3].split(":")[1]
        start, end = coords.split('-')
        start = int(start)
        end = int(end)
        copy = int(line[4])
        while True:
            line = f.readline().strip()
            line = line.split('\t')
            copy = int(line[4])
            if copy > 1:
                break
            coords = line[3].split(":")[1]
        _, final = coords.split('-')
        final = int(final)
    print(start, end, final)

    # create coords dict
    coords_dict = {} # coords : ind
    i=0
    while end <= final:
        coords_dict[f'chr{chr}:{start}-{end}'] = i
        i += 1
        start = end
        end += 50000
    num_coords = len(coords_dict)
    with open(coords_file, 'w') as f:
        json.dump(coords_dict, f)

    # create xyz_file
    if not osp.exists(xyz_file):
        # count number of cells
        all_numbers = set()
        with open(file, 'r') as f:
            col_names = f.readline()
            reader = csv.reader(f, delimiter = '\t')
            for line in reader:
                coords = line[3]
                cell = int(line[4])
                all_numbers.add(cell)

        num_cells = len(all_numbers)
        assert num_cells == sorted(all_numbers)[-1]
        print(f'found {num_coords} coords and {num_cells} cells')


        AC = ArgparserConverter()
        xyz = np.empty((num_cells, num_coords, 3))
        xyz[:] = np.NaN
        with open(file, 'r') as f:
            col_names = f.readline()
            reader = csv.reader(f, delimiter = '\t')
            for line in reader:
                z, x, y, coords, cell = line[:5]
                x, y, z, cell = [AC.str2int(i) for i in [x, y, z, cell]]
                cell -= 1 # convert to 0-based indexing
                xyz[cell, coords_dict[coords], :] = [x, y, z]

        np.save(xyz_file, xyz)
    else:
        xyz = np.load(xyz_file)

    print(xyz.shape, xyz.dtype, file = log_file)
    num_cells, num_coords, _ = xyz.shape

    D = xyz_to_distance(xyz, True)
    np.save(osp.join(dir, 'dist.npy'), D)

    D_prox = to_proximity(D)
    print(D_prox, file = log_file)
    print('mean', np.nanmean(np.diagonal(D_prox, 1)), file = log_file)
    np.save(osp.join(dir, 'dist_proximity.npy'), D_prox)


    D_mean = np.nanmean(D, axis = 0)
    print(D_mean, file = log_file)
    print('mean', np.nanmean(np.diagonal(D_mean, 1)), file = log_file)
    np.save(osp.join(dir, 'dist_mean.npy'), D_mean)

    D_median = np.nanmedian(D, axis = 0)
    print(D_median, file = log_file)
    print('mean', np.nanmean(np.diagonal(D_median, 1)), file = log_file)
    print('median', np.nanmean(np.diagonal(D_median, 1)), file = log_file)

    np.save(osp.join(dir, 'dist_median.npy'), D_median)
    #
    # # create negative control
    # D1 = D[:len(D)//2]
    # D1_mean = np.nanmean(D1, axis = 0)
    # np.save(osp.join(dir, 'dist_mean_first_half.npy'), D1_mean)
    #
    # D2 = D[len(D)//2:]
    # D2_mean = np.nanmean(D2, axis = 0)
    # np.save(osp.join(dir, 'dist_mean_second_half.npy'), D2_mean)

    log_file.close()

def xyz_to_xyz():
    '''Convert to proper xyz format'''
    chr=21
    dir = '/home/erschultz/Su2020/samples/sample1'
    xyz_file = osp.join(dir, 'xyz.npy')
    xyz = np.load(xyz_file)
    # xyz_write(xyz, osp.join(dir, 'xyz.xyz'), 'w')

    n = 100
    D = xyz_to_distance(xyz[:n])
    print(D.shape)
    fig, ax = plt.subplots()
    for i in range(n):
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(D[i, :, :])
        ax.plot(meanDist, label = i)
    ax.legend(loc='upper left')
    # ax.set_xscale('log')
    ax.set_ylabel('Distance (nm)', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'dist_scaling_per_frame.png'))
    plt.close()

    D_mean = np.load(osp.join(dir, 'dist_mean.npy'))
    print(D_mean, D_mean.shape)
    D_std = np.nanstd(D, axis = 0, ddof=1)
    print(D_std, D_std.shape)
    fig, ax = plt.subplots()
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_mean)
    meanDist_std = DiagonalPreprocessing.genomic_distance_statistics(D_mean, stat='std')

    x = np.arange(0, len(meanDist))
    ax.plot(x, meanDist, label = i, color = 'b')
    ax.fill_between(x, meanDist - meanDist_std, meanDist + meanDist_std, color = 'b', alpha = 0.5)
    # ax.set_xscale('log')
    ax.set_ylabel('Distance (nm)', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'dist_scaling_avg.png'))
    plt.close()

def sim_xyz_to_dist(dir, max_ent=True):
    if max_ent:
        final_dir = get_final_max_ent_folder(dir)
        file = osp.join(final_dir, 'production_out/output.xyz')
    else:
        file = osp.join(dir, 'production_out/output.xyz')

    xyz = xyz_load(file, multiple_timesteps = True, N_min = 5)
    D = xyz_to_distance(xyz)

    D_mean = np.nanmean(D, axis = 0)
    # print(D_mean)
    # print('mean', np.nanmean(np.diagonal(D_mean, 1)))
    np.save(osp.join(dir, 'dist_mean.npy'), D_mean)

    D_median = np.nanmedian(D, axis = 0)
    # print(D_mean)
    # print('mean', np.nanmean(np.diagonal(D_mean, 1)))
    np.save(osp.join(dir, 'dist_median.npy'), D_median)

    return D_mean, D_median

def controls():
    # deprecated
    def compare_control():
        dir = '/home/erschultz/Su2020/samples/sample1'
        D_file = osp.join(dir, 'dist_mean_first_half.npy')
        D1 = np.load(D_file)
        m = 512
        D1 = crop_hg38(D1, 'chr21:14000001-14050001', m)
        print(D1)


        D_file = osp.join(dir, 'dist_mean_second_half.npy')
        D2 = np.load(D_file)
        D2 = crop_hg38(D2, 'chr21:14000001-14050001', m)
        print(D2)

        triu_ind = np.triu_indices(len(D1))
        overall_corr = pearson_round(D1[triu_ind], D2[triu_ind], stat = 'nan_pearson')
        overall_corr2 = pearson_round(D1, D2, stat = 'nan_pearson')
        print(overall_corr, overall_corr2)


        meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(D1)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(D2)
        plot_mean_dist(meanDist, dir, 'dist_meanDist_log_ref.png',
                        None, True, True, meanDist_gt, 'Exp 1/2', 'Exp 2/2',
                        'blue', ylabel='Distance (nm)')
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(D2)


        # scc = SCC()
        # corr_scc = scc.scc(D, D_sim, var_stabilized = False)
        # corr_scc_var = scc.scc(D, D_sim, var_stabilized = True)
        avg_diag, corr_arr = calc_dist_strat_corr(D1, D2, mode = 'nan_pearson',
                                                return_arr = True)
        avg_diag = np.round(avg_diag, 3)

        # format title
        title = f'Overall Pearson Corr: {overall_corr}'
        title += f'\nMean Diagonal Pearson Corr: {avg_diag}'
        # title += f'\nSCC: {corr_scc_var}'

        log=True
        plt.plot(np.arange(m-2), corr_arr, color = 'black')
        plt.ylim(-0.5, 1)
        plt.xlabel('Distance', fontsize = 16)
        plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
        plt.title(title, fontsize = 16)

        plt.tight_layout()
        if log:
            plt.xscale('log')
            plt.savefig(osp.join(dir, 'dist_distance_pearson_log.png'))
        else:
            plt.savefig(osp.join(dir, 'dist_distance_pearson.png'))
        plt.close()

    def compare_sim_Hic_to_sim_D():
        dir = '/home/erschultz/dataset_test/samples/sample5000'
        max_ent_dir = osp.join(dir, 'soren-S/k10/replicate1')
        # max_ent_dir = '/home/erschultz/dataset_bond/samples/sample9'
        max_ent = True
        D = sim_xyz_to_dist(max_ent_dir, max_ent)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(D)

        Y, Y_diag = load_Y(max_ent_dir)

        V_D = get_pcs(D)
        V_Y = get_pcs(Y)

        rows = 3; cols = 3
        row = 0; col = 0
        fig, ax = plt.subplots(rows, cols)
        fig.set_figheight(12)
        fig.set_figwidth(16)
        for i in range(rows*cols):
            ax[row,col].plot(V_D[i], label = 'simulated dist')
            ax[row,col].plot(V_Y[i], label = 'simulated Hi-C')
            ax[row,col].set_title(f'PC {i+1}')
            ax[row,col].legend(fontsize=16)

            col += 1
            if col > cols-1:
                col = 0
                row += 1
        plt.savefig(osp.join(max_ent_dir, 'pc_D_vs_Y.png'))

def compare_D_to_sim_D(sample, GNN_ID=None):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID)
    D, _, _ = load_exp_gnn_pca(dir, GNN_ID)
    # max_ent_dir = '/home/erschultz/dataset_bond/samples/sample9'
    if GNN_ID is not None:
        max_ent = False
    else:
        max_ent = True
    if max_ent:
        sim_dir = max_ent_dir
        config_file = osp.join(sim_dir, 'resources/config.json')
    else:
        sim_dir = gnn_dir
        config_file = osp.join(sim_dir, 'config.json')
    D_sim, _ = sim_xyz_to_dist(sim_dir, max_ent)
    m = len(D_sim)

    # load config params
    with open(config_file, 'r') as f:
        config = json.load(f)
        bl = config["bond_length"]
        delta = config["grid_size"]
        vb = config['beadvol']

    # process experimental distance map
    nan_rows = np.isnan(D[0])
    D_no_nan = D[~nan_rows][:, ~nan_rows]
    plot_distance_map(D, sim_dir, 'exp')
    plot_distance_map(D_sim, sim_dir, 'sim')

    triu_ind = np.triu_indices(len(D))
    overall_corr = pearson_round(D[triu_ind], D_sim[triu_ind], stat = 'nan_pearson')
    overall_corr2 = pearson_round(D, D_sim, stat = 'nan_pearson')
    print(overall_corr, overall_corr2)


    epilib.eric_plot_tri(D_sim, D, osp.join(sim_dir, 'dist_triu.png'), vmaxp = np.nanmax(D),
            title = f'Corr = {overall_corr}', cmap = BLUE_CMAP)

    title = f'b={bl}, gs={delta}, vb={vb}'
    meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(D)
    # print(meanDist_gt)
    np.save(osp.join(sim_dir, 'meanDist_gt.npy'), meanDist_gt)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_sim)
    plot_mean_dist(meanDist, sim_dir, 'dist_meanDist_log_ref.png',
                    None, True, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', title, 'Distance (nm)')
    plot_mean_dist(meanDist, sim_dir, 'dist_meanDist_ref.png',
                    None, False, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', title, 'Distance (nm)')
    plot_mean_dist(meanDist, sim_dir, 'dist_meanDist_ref.png',
                    None, False, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', title, 'Distance (nm)')


    # scc = SCC()
    # corr_scc = scc.scc(D, D_sim, var_stabilized = False)
    # corr_scc_var = scc.scc(D, D_sim, var_stabilized = True)
    avg_diag, corr_arr = calc_dist_strat_corr(D_no_nan, D_sim[~nan_rows][:, ~nan_rows],
                                            mode = 'nan_pearson', return_arr = True)
    avg_diag = np.round(avg_diag, 3)

    # format title
    title = f'Overall Pearson Corr: {overall_corr}'
    title += f'\nMean Diagonal Pearson Corr: {avg_diag}'
    # title += f'\nSCC: {corr_scc_var}'

    log=True
    plt.plot(corr_arr, color = 'black')
    plt.ylim(-0.5, 1)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
    plt.title(title, fontsize = 16)

    plt.tight_layout()
    if log:
        plt.xscale('log')
        plt.savefig(osp.join(sim_dir, 'dist_distance_pearson_log.png'))
    else:
        plt.savefig(osp.join(sim_dir, 'dist_distance_pearson.png'))
    plt.close()

    # plot_diagonal(D, D_sim)


    # compare PCs
    V_D = get_pcs(D, nan_rows)
    # plt.plot(V[0])
    # plt.show()
    V_D_sim = get_pcs(D_sim, nan_rows)


    rows = 2; cols = 2
    row = 0; col = 0
    fig, ax = plt.subplots(rows, cols)
    fig.set_figheight(12)
    fig.set_figwidth(16)
    for i in range(rows*cols):
        ax[row,col].plot(V_D[i], label = 'experimental dist')
        ax[row,col].plot(V_D_sim[i], label = 'simulated dist')
        ax[row,col].set_title(f'PC {i+1}\nCorr={pearson_round(V_D[i], V_D_sim[i])}')
        ax[row,col].legend(fontsize=16)

        col += 1
        if col > cols-1:
            col = 0
            row += 1
    plt.savefig(osp.join(sim_dir, 'pc_D_vs_D_sim.png'))

def get_dirs(dir, GNN_ID, b=140, phi=0.03):
    max_ent_dir = osp.join(dir, f'optimize_grid_b_{b}_phi_{phi}-max_ent10')
    if GNN_ID is not None:
        gnn_dir = osp.join(dir, f'optimize_grid_b_{b}_phi_{phi}-GNN{GNN_ID}')
    else:
        gnn_dir = None
    return max_ent_dir, gnn_dir

def load_exp_gnn_pca(dir, GNN_ID=None, mode='mean', b=140, phi=0.03):
    result = load_import_log(dir)
    start = result['start']
    resolution = result['resolution']
    chrom = int(result['chrom'])
    genome = result['genome']

    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi)
    if osp.exists(max_ent_dir):
        D_pca, _ = sim_xyz_to_dist(max_ent_dir, True)
        m = len(D_pca)
    else:
        print(f'{max_ent_dir} does not exist')
        D_pca = None

    if GNN_ID is not None and osp.exists(gnn_dir):
        D_gnn, _ = sim_xyz_to_dist(gnn_dir, False)
        m = len(D_gnn)
    else:
        D_gnn = None

    # process experimental distance map
    if genome == 'hg38':
        if chrom == 21:
            exp_dir = '/home/erschultz/Su2020/samples/sample1'
        elif chrom == 2:
            exp_dir = '/home/erschultz/Su2020/samples/sample10'
        else:
            raise Exception(f'Unrecognized chrom: {chrom}')
        D_file = osp.join(exp_dir, 'dist_mean.npy')
        D = np.load(D_file)
        D = crop_hg38(exp_dir, D, f'chr{chrom}:{start}-{start+resolution}', m)
        np.save(osp.join(dir, 'D_crop.npy'), D)
    else:
        D = None

    return D, D_gnn, D_pca

def compare_pcs(sample, GNN_ID):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID)
    nan_rows = np.isnan(D[0])

    # compare PCs
    V_D = get_pcs(D, nan_rows, True)
    V_D_pca = get_pcs(D_pca, nan_rows)
    V_D_gnn = get_pcs(D_gnn, nan_rows)


    rows = 2; cols = 1
    row = 0; col = 0
    fig, ax = plt.subplots(rows, cols)
    fig.set_figheight(12)
    fig.set_figwidth(16)
    for i in range(rows*cols):
        ax[row].plot(V_D[i], label = 'Experiment', color = 'k')
        if V_D_pca is not None:
            ax[row].plot(V_D_pca[i], label = 'Max. Ent.', color = 'blue')
        if V_D_gnn is not None:
            ax[row].plot(V_D_gnn[i], label = 'GNN', color = 'red')
        ax[row].set_title(f'PC {i+1}')
        if i == 0:
            ax[row].legend(fontsize=16)

        col += 1
        if col > cols-1:
            col = 0
            row += 1
    plt.savefig(osp.join(dir, 'pc_D_vs_D_sim.png'))
    plt.close()

def compare_d_maps(sample, GNN_ID):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID)
    nan_rows = np.isnan(D[0])
    D_list = [D_gnn, D_pca, D]
    labels = ['GNN', 'Max Ent', 'Experiment']



    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,1]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    vmax = np.nanmean(D)
    for i, (D_i, label) in enumerate(zip(D_list, labels)):
        if D_i is None:
            continue
        D_i = D_i[~nan_rows][:, ~nan_rows] # ignore nan_rows
        triu_ind = np.triu_indices(len(D_i))

        # if i == 2:
        #     s = sns.heatmap(D_i, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_CMAP,
        #                     ax = ax[i], cbar_ax = ax[3])
        # else:
        s = sns.heatmap(D_i, linewidth = 0, vmin = 0, vmax = np.nanmean(D_i), cmap = BLUE_CMAP,
                        ax = ax[i], cbar = False)

        corr = pearson_round(D[~nan_rows][:, ~nan_rows][triu_ind], D_i[triu_ind], stat = 'nan_pearson')
        title = (f'{label}'
                f'\nCorr={np.round(corr, 3)}')
        s.set_title(title, fontsize = 16)

        if i > 0:
            s.set_yticks([])


    plt.tight_layout()
    plt.savefig(osp.join(dir, 'D_PCA_vs_GNN.png'))
    plt.close()

def compare_rg(sample, GNN_ID, b=140, phi=0.03):
    label_fontsize=18
    legend_fontsize=16
    tick_fontsize=13
    letter_fontsize=20
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'

    result = load_import_log(dir)
    start = result['start']
    end = result['end']
    start_mb = result['start_mb']
    end_mb = result['end_mb']
    chrom = int(result['chrom'])
    resolution = result['resolution']

    # distance distribution
    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    coords_a = f"chr{chrom}:15500001-15550001"
    coords_b = f"chr{chrom}:20000001-20050001"
    coords_a_label =  f"chr{chrom}:15.5 -15.55 Mb"
    coords_b_label = f"chr{chrom}:20-20.05 Mb"
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz = np.load(xyz_file)
    coords_file = osp.join(exp_dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)

    a = coords_dict[coords_a]
    b = coords_dict[coords_b]
    dist = dist_distribution_a_b(xyz, a, b)

    # shift ind such that start is at 0
    shift = coords_dict[f"chr{chrom}:{start}-{start+resolution}"]

    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi)
    final_dir = get_final_max_ent_folder(max_ent_dir)
    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_max_ent = xyz_load(file, multiple_timesteps = True)

    if gnn_dir is not None and osp.exists(gnn_dir):
        file = osp.join(gnn_dir, 'production_out/output.xyz')
        xyz_gnn = xyz_load(file, multiple_timesteps = True)
    else:
        xyz_gnn = None

    sizes = [16, 32, 64, 128, 256, 512]
    log_labels = [i*resolution for i in sizes]
    ref_rgs = np.zeros((len(sizes), 2))
    max_ent_rgs = np.zeros((len(sizes), 2))
    gnn_rgs = np.zeros((len(sizes), 2))
    for i, size in enumerate(sizes):
        left = int(256 - size/2)
        right = int(256 + size/2)
        xyz_size = xyz[:, left:right, :]
        xyz_max_ent_size = xyz_max_ent[:, left:right, :]
        xyz_gnn_size = xyz_gnn[:, left:right, :]

        ref_rgs[i] = calculate_rg(xyz_size, verbose = False)
        max_ent_rgs[i] = calculate_rg(xyz_max_ent_size)
        gnn_rgs[i] = calculate_rg(xyz_gnn_size)


    print(ref_rgs[:, 0])
    plt.errorbar(log_labels, ref_rgs[:, 0], ref_rgs[:, 1], color = 'k', label = 'Experiment')
    plt.errorbar(log_labels, max_ent_rgs[:, 0], max_ent_rgs[:, 1], color = 'b', label = 'Max Ent')
    plt.errorbar(log_labels, gnn_rgs[:, 0], gnn_rgs[:, 1], color = 'r', label = 'GNN')

    X = np.linspace(log_labels[0], log_labels[-1], 100)
    Y = np.power(X, 1/4)
    Y = Y * ref_rgs[0, 0] / np.min(Y)
    plt.plot(X, Y, label = '1/4', ls='dashed', color = 'gray')
    Y = np.power(X, 1/3)
    Y = Y * ref_rgs[0, 0] / np.min(Y)
    plt.plot(X, Y, label = '1/3', ls='dotted', color = 'gray')

    plt.ylabel('Radius of Gyration', fontsize=16)
    plt.xlabel('Domain Size (bp)', fontsize=16)
    plt.legend(fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'rg_comparison.png'))

def compare_scaling(sample, GNN_ID=None, b=140, phi=0.03):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID, b=b, phi=phi)
    print(D_pca)
    D_list = [D, D_pca, D_gnn]
    labels = ['Experiment', 'Max Ent', 'GNN']
    colors = ['k', 'b', 'r']

    for D_i, label in zip(D_list, labels):
        if D_i is not None:
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_i, 'freq')
            plt.plot(meanDist, label = label)

    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'dist_scaling.png'))
    plt.close()

    for D_i, label in zip(D_list, labels):
        if D_i is not None:
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_i, 'freq')
            plt.plot(meanDist, label = label)

    plt.xscale('log')
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'dist_scaling_log.png'))
    plt.close()

def compare_dist_distribution_plaid(sample, GNN_ID, b=140, phi=0.03):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    y = np.load(osp.join(dir, 'y.npy'))
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(epilib.get_oe(y))
    seq_a = np.array(kmeans.labels_, dtype=bool)
    seq_b = ~seq_a

    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi)
    final_dir = get_final_max_ent_folder(max_ent_dir)

    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_max_ent = xyz_load(file, multiple_timesteps = True)
    D_pca = xyz_to_distance(xyz_max_ent, False)


    if gnn_dir is not None:
        file = osp.join(gnn_dir, 'production_out/output.xyz')
        xyz_gnn = xyz_load(file, multiple_timesteps = True)
        D_gnn = xyz_to_distance(xyz_gnn, False)
    else:
        D_gnn = None


    result = load_import_log(dir)
    chrom = int(result['chrom'])
    start = result['start']
    resolution = result['resolution']
    # process experimental distance map
    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    else:
        raise Exception(f'Unrecognized chrom: {chrom}')
    d_file = osp.join(exp_dir, 'dist.npy')
    D = np.load(d_file)
    D = crop_hg38(exp_dir, D, f'chr{chrom}:{start}-{start+resolution}', len(y))

    fig, ax = plt.subplots(1, 3)
    data = zip([D, D_max_ent, D_gnn], ['Experiment', 'Max Ent', 'GNN'])
    for i, (D_i, method) in enumerate(data):
        print(method)
        if D_i is None:
            continue
        results = dist_distribution_seq(D_i, seq_a, seq_b) #all_aa, all_bb, all_ab

        bin_width = 50
        for arr, label in zip(results, ['A-A', 'B-B', 'A-B']):
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            print(np.round(mean, 1), '+-', np.round(std, 1))
            ax[i].hist(arr, label = label, alpha=0.5,
                weights = np.ones_like(arr) / len(arr),
                bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width))

        # ax[i].set_xscale('log')
        ax[i].set_title(method)
        ax[i].legend(fontsize=16)

    fig.supxlabel('Distance (nm)')
    fig.supylabel('Probability')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'dist_distribution_plaid.png'))
    plt.close()

def compare_diagonal(sample, GNN_ID=None):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi)
    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID)
    np.save(osp.join(D, 'D.npy'), D)
    np.save(osp.join(D, 'D_gnn.npy'), D_gnn)
    np.save(osp.join(D, 'D_pca.npy'), D_pca)

    m = len(D_pca)

    # process experimental distance map
    nan_rows = np.isnan(D[0])
    D_no_nan = D[~nan_rows][:, ~nan_rows]

    plot_diagonal(D_no_nan, D_pca[~nan_rows][:, ~nan_rows], osp.join(dir, 'diagonal.png'))

def compare_dist_ij(sample, GNN_ID=None, b=140, phi=0.03):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID, b=b, phi=phi)
    nan_rows = np.isnan(D[0])
    D_no_nan = D[~nan_rows][:, ~nan_rows] # ignore nan_rows
    D_pca_no_nan = D_pca[~nan_rows][:, ~nan_rows]

    result = load_import_log(dir)
    start = result['start']
    end = result['end']
    start_mb = result['start_mb']
    end_mb = result['end_mb']
    chrom = int(result['chrom'])
    resolution = result['resolution']

    m = len(D[~nan_rows][:, ~nan_rows])
    all_labels = np.linspace(start_mb, end_mb, m)
    all_labels = np.round(all_labels, 1)
    genome_ticks = [0, m-1]
    genome_labels = [f'{all_labels[i]} Mb' for i in genome_ticks]

    print(D)
    D_flat = D_no_nan.flatten()
    print(D_flat)
    plt.scatter(D_no_nan.flatten(), D_pca_no_nan.flatten())
    plt.axline((0,0), slope=1, color = 'k')
    plt.xlabel(r'$D_{ij}$', fontsize=16)
    plt.ylabel(r'$D^{PCA}_{ij}$', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # tsv_to_npy()
    # test_pcs()
    # load_exp_gnn_pca('/home/erschultz/Su2020/samples/sample1002')
    # find_hg38_positions()
    # xyz_to_dist()
    # xyz_to_xyz()
    # compare_D_to_sim_D(1014)
    compare_diagonal(1013)
    # sim_xyz_to_dist('/home/erschultz/Su2020/samples/sample1011/optimize_grid_b_140_phi_0.03-GNN403', False)
    # find_volume()
    # compare_pcs(1013)
    # compare_d_maps(1003, None)
    # compare_dist_distribution_a_b()
    # compare_dist_distribution_plaid(1013, None, 261, 0.01)
    # compare_rg(1014, 423, b=261, phi=0.01)
    # compare_scaling(1002, None, 261, 0.006)
    # compare_dist_ij(1014, 423, b=261, phi=0.01)
