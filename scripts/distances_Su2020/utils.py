import csv
import json
import math
import os
import os.path as osp
import string
import sys

import liftover
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import seaborn as sns
from pylib.utils import epilib
from pylib.utils.ArgparseConverter import ArgparseConverter
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.load_utils import get_final_max_ent_folder, load_Y
from pylib.utils.plotting_utils import (BLUE_CMAP, BLUE_RED_CMAP,
                                        RED_BLUE_CMAP, RED_CMAP,
                                        calc_dist_strat_corr, plot_matrix,
                                        plot_mean_dist, rotate_bound)
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import load_import_log, nan_pearsonr, pearson_round
from pylib.utils.xyz import (calculate_rg, calculate_rg_matrix, xyz_load,
                             xyz_to_distance, xyz_write)
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


# load data
def load_exp_gnn_pca(dir, GNN_ID=None, b=180, phi=None, v=8, ar=1.5, mode='mean'):
    result = load_import_log(dir)
    start = result['start']
    end = result['end']
    resolution = result['resolution']
    chrom = result['chrom']
    genome = result['genome']

    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi, v, ar)
    if osp.exists(max_ent_dir):
        D_pca, _ = sim_xyz_to_dist(max_ent_dir, True)
        m = len(D_pca)
    else:
        print(f'ME: {max_ent_dir} does not exist')
        D_pca = None
        D_med_pca = None

    if GNN_ID is not None and osp.exists(gnn_dir):
        D_gnn, _ = sim_xyz_to_dist(gnn_dir, False)
        m = len(D_gnn)
    else:
        print(f'GNN: {gnn_dir} does not exist')
        D_gnn = None
        D_med_gnn = None

    # process experimental distance map
    data_dir = os.sep.join(dir.split(os.sep)[:-2])
    D = np.load(osp.join(data_dir, f'dist_mean_chr{chrom}.npy'))
    with open(osp.join(data_dir, f'coords_chr{chrom}.json')) as f:
        coords_dict = json.load(f)
    if genome == 'hg38':
        D = crop_hg38(D, f'chr{chrom}:{start}-{start+resolution}', m, coords_dict)
    elif genome == 'hg19':
        converter = liftover.get_lifter('hg19', 'hg38')
        start_hg38 = converter[chrom][start][0][1]
        end_hg38 = converter[chrom][end][0][1]
        m_hg38 = int(end_hg38 - start_hg38) / resolution

        start_hg38 = start_hg38 // resolution * resolution + 1
        end_hg38 = end_hg38 // resolution * resolution + 1
        m_hg38 = int((end_hg38 - start_hg38) / resolution)
        D = crop_hg38(D, f'chr{chrom}:{start_hg38}-{start_hg38+resolution}', m_hg38, coords_dict)
    else:
        raise Exception()

    np.save(osp.join(dir, 'D_crop.npy'), D)
    assert mode == 'mean', 'not implemented'
    return D, D_gnn, D_pca

# plotting functions
def plot_diagonal(exp, sim, ofile=None, height=30):
    npixels = np.shape(sim)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)

    # make composite contact map
    composite = np.zeros((npixels, npixels))
    composite[indu] = sim[indu]
    composite[indl] = exp[indl]
    print('orig', composite.shape)
    m = len(composite)

    # Rotate 45 degs
    resized = rotate_bound(composite,-45)
    print('resize', resized.shape)

    # crop
    center = resized.shape[0] // 2
    m_resize = len(resized)
    resized = resized[center-height:center+height, :]


    vmin = np.nanpercentile(exp, 1)
    vmax = np.nanpercentile(exp, 99)
    fig, ax = plt.subplots(figsize=(6, 2),dpi=600)
    # _pf = ax.imshow(resized, vmin=200, vmax=1000, cmap='seismic')
    sns.heatmap(resized, ax = ax, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP)
    ax.set_xticks([0, m_resize/4, m_resize/2, m_resize*3/4, m_resize],
                labels = np.round([0, m/4, m/2, m*3/4, m], 0).astype(int))
    ax.set_yticks([0, height, height*2],
                labels = np.round([m*height/m_resize, 0, m*height/m_resize], 0).astype(int))

    plt.tight_layout()

    if ofile is not None:
        plt.savefig(ofile)
        plt.close()
    else:
        plt.show()

def plot_distance_map(input, dir, label):
    plot_matrix(input, osp.join(dir, f'D_{label}.png'), f'D_{label}', vmax = 'max',
                cmap=RED_BLUE_CMAP)
    # normalize by genomic distance
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(input)
    input_diag = DiagonalPreprocessing.process(input, meanDist)

    plot_matrix(input_diag, osp.join(dir, f'D_diag_{label}.png'), f'D_diag_{label}',
                vmin = 'center1', cmap='bluered')

    if np.sum(np.isnan(input_diag)) == 0:
        input_diag_corr = np.corrcoef(input_diag)
        plot_matrix(input_diag_corr, osp.join(dir, f'D_diag_corr_{label}.png'),
                    f'D_diag_corr_{label}', vmin = -1, vmax= 1, cmap='bluered')

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

def crop_hg38(inp, start, m, coords):
    '''
    Crop input distance map to m pixels starting from start.

    Inputs:
        inp (np array): distance map
        start (str): genomic coordinates
        m (int): desired len of output
        coords: dictionary
    '''
    assert len(inp) == len(coords)
    i = coords[start]
    if len(inp.shape) == 2:
        return inp[i:i+m, i:i+m]
    elif len(inp.shape) == 3:
        return inp[:, i:i+m, i:i+m]
    else:
        raise Exception(f'Unaccepted shape: {inp.shape}')

def min_MSE(D, D_sim):
    def metric(alpha, D, D_sim):
        return mean_squared_error(D, D_sim*alpha)

    result = minimize(metric, x0=1, args=(D, D_sim))
    print(f'resultt {result}')


    return result.x

def get_pcs(input, nan_rows=None, verbose=False, smooth=False, h=1,
            return_input=False):
    if input is None:
        return None
    if nan_rows is None:
        nan_rows = np.isnan(input[0])
    input = input[~nan_rows][:, ~nan_rows] # ignore nan_rows

    if smooth:
        input = scipy.ndimage.gaussian_filter(input, (h, h))

    input = epilib.get_oe(input)
    # input = np.corrcoef(input)

    if verbose:
        print(input)
    seqs = epilib.get_pcs(input, 12,
                        normalize = False, align = True)
    if return_input:
        return seqs.T, input
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
def find_volume():
    '''Find average volume of cells in dataset.'''
    dir = '/home/erschultz/Su2020/samples'
    odir = '/home/erschultz/TICG-chromatin/figures/distances'
    s_dir = osp.join(dir, 'sample1013_rescale1')
    b = 200; v = 8; ar=1.5
    xyz_file = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}_spheroid_{ar}-max_ent10',
                        'iteration30/production_out/output.xyz')
    xyz_sim = xyz_load(xyz_file, multiple_timesteps = True)
    N, m, _ = xyz_sim.shape
    result = load_import_log(s_dir)
    start = result['start']
    resolution = result['resolution']
    chrom = int(result['chrom'])
    genome = result['genome']

    if genome == 'hg38':
        if chrom == 21:
            exp_dir = osp.join(dir, 'sample1')
        elif chrom == 2:
            exp_dir = osp.join(dir, 'sample10')
        else:
            raise Exception(f'Unrecognized chrom: {chrom}')
    coords_file = osp.join(exp_dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)

    i = coords_dict[f'chr{chrom}:{start}-{start+resolution}']
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz_exp = np.load(xyz_file)
    xyz_exp = xyz_exp[:, i:i+m, :]

    def get_vols(xyz):
        print('xyz', xyz.shape, xyz.dtype)
        D = xyz_to_distance(xyz)
        num_cells, num_coords, _ = xyz.shape
        vols = np.zeros(num_cells)
        r_spheres = np.zeros(num_cells)
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

            D_i = D[i]
            r_sphere = np.nanmax(D_i)
            # this is an approximate solution to the bounding spheres problem
            # it lower bounds the true solution
            r_spheres[i] = r_sphere

        mean_vol = np.nanmean(vols)
        median_vol = np.nanmedian(vols)
        std_vol = np.nanstd(vols)
        max_vol = np.nanmax(vols)
        print(f'median vol {mean_vol} um^3')
        print(f'mean vol {median_vol} um^3')
        print(f'std vol {std_vol}')
        print(f'max vol {max_vol}')
        mean_r_sphere = np.nanmean(r_spheres) / 2
        print(f'mean spherical radius {r_sphere} nm')
        return vols, r_spheres

    print('---Experiment---')
    exp_hull, exp_radii = get_vols(xyz_exp)
    arr = exp_hull; bin_width = 1
    plt.hist(arr, alpha=0.5, label = 'Experiment',
                weights = np.ones_like(arr) / len(arr),
                bins = np.arange(math.floor(min(arr)),
                            math.ceil(max(arr)) + bin_width,
                            bin_width))

    print('---Simulation---')
    sim_hull, sim_radii = get_vols(xyz_sim)
    arr = sim_hull
    plt.hist(arr, alpha=0.5, label = 'Simulation',
                weights = np.ones_like(arr) / len(arr),
                bins = np.arange(math.floor(min(arr)),
                            math.ceil(max(arr)) + bin_width,
                            bin_width))

    plt.xlim(None, np.nanpercentile(exp_hull, 99))
    plt.title(f'b={b}, v={v}')
    plt.xlabel(r'Volume $\mu m^3$')
    plt.ylabel('Probability')
    plt.xlabel('Convex Hull of Structure')
    plt.legend()
    plt.savefig(osp.join(odir, 'volumes.png'))
    plt.close()

    # bin_width = 100
    # for arr, label in zip([vols_arr1, vols_arr2], ['Experiment', 'Simulation']):
    #     plt.hist(arr, alpha=0.5, label = label,
    #                 weights = np.ones_like(arr) / len(arr),
    #                 bins = np.arange(math.floor(min(arr)),
    #                             math.ceil(max(arr)) + bin_width,
    #                             bin_width))
    # plt.title(f'b={b}, phi={phi}')
    # plt.xlabel(r'Spherical Radius nm')
    # plt.ylabel('Probability')
    # plt.legend()
    # plt.savefig(osp.join(odir, 'spherical_radii.png'))
    # plt.close()

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
                bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width,
                            bin_width))
    arr = dist_sim
    plt.hist(arr, label = 'Max. Ent.', alpha = 0.5, color = 'blue',
                weights = np.ones_like(arr) / len(arr),
                bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width,
                            bin_width))
    plt.xlabel('Distance between A and B', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.legend(fontsize=16)
    plt.title(f'Distance between\n{coords_a} and {coords_b}')
    # plt.title(f'Distance between A and B')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'distance_distribution.png'))
    plt.close()

def xyz_to_dist2():
    '''Convert experimental xyz to distance map.'''
    chr=21
    dir = '/home/erschultz/Su2020/Bintu/K562'
    log_file = osp.join(dir, 'xyz_to_dist.log')
    log_file = open(log_file, 'w')
    file = osp.join(dir, f'chromosome{chr}-28-30Mb.csv')
    print('ifile: ', file)

    num_coords = 65

    # create xyz_file
    # count number of cells
    all_numbers = set()
    with open(file, 'r') as f:
        f.readline(); f.readline()
        reader = csv.reader(f)
        for line in reader:
            cell = int(line[0])
            all_numbers.add(cell)

    num_cells = len(all_numbers)
    assert num_cells == sorted(all_numbers)[-1]
    print(f'found {num_coords} coords and {num_cells} cells')


    AC = ArgparseConverter()
    xyz = np.empty((num_cells, num_coords, 3))
    xyz[:] = np.NaN
    with open(file, 'r') as f:
        f.readline(); f.readline()
        reader = csv.reader(f)
        for line in reader:
            cell, i, z, x, y = line
            cell, i, z, x, y = [AC.str2int(j) for j in [cell, i, z, x, y]]

            # convert to 0-based indexing
            i -= 1
            cell -= 1

            xyz[cell, i, :] = [x, y, z]
    xyz_file = osp.join(dir, 'xyz2.npy')
    np.save(xyz_file, xyz)

    print(xyz[0])
    print(xyz.shape, xyz.dtype, file = log_file)
    num_cells, num_coords, _ = xyz.shape

    D = xyz_to_distance(xyz, False)
    np.save(osp.join(dir, 'dist2.npy'), D)

    D_mean = np.nanmean(D, axis = 0)
    print(D_mean, file = log_file)
    print('mean', np.nanmean(np.diagonal(D_mean, 1)), file = log_file)
    np.save(osp.join(dir, 'dist2_mean.npy'), D_mean)

    log_file.close()

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


        AC = ArgparseConverter()
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
    ax.fill_between(x, meanDist - meanDist_std, meanDist + meanDist_std,
                    color = 'b', alpha = 0.5)
    # ax.set_xscale('log')
    ax.set_ylabel('Distance (nm)', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'dist_scaling_avg.png'))
    plt.close()

def sim_xyz_to_dist(dir, max_ent=True):
    if max_ent:
        final_dir = get_final_max_ent_folder(dir)
    else:
        final_dir = dir
    if osp.exists(osp.join(final_dir, 'production_out')):
        file = osp.join(final_dir, 'production_out/output.xyz')
    else:
        file = osp.join(final_dir, 'output.xyz')

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

def compare_D_to_sim_D(sample, GNN_ID=None, b=180, phi=None, v=8, ar=1.5):
    dir = f'/home/erschultz/dataset_11_20_23/samples/sample{sample}'
    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi, v, ar)
    D, D_gnn, D_sim = load_exp_gnn_pca(dir, GNN_ID)

    m = len(D_sim)

    # process experimental distance map
    nans = np.isnan(D)
    nan_rows = np.zeros(m).astype(bool)
    nan_rows[np.sum(nans, axis=0) == m] = True
    D_no_nan = D[~nan_rows][:, ~nan_rows]
    plot_distance_map(D, max_ent_dir, 'exp')
    plot_distance_map(D_no_nan, max_ent_dir, 'exp_no_nan')
    plot_distance_map(D_sim, max_ent_dir, 'sim')

    triu_ind = np.triu_indices(len(D))
    overall_corr = pearson_round(D[triu_ind], D_sim[triu_ind], stat = 'nan_pearson')
    overall_corr2 = pearson_round(D, D_sim, stat = 'nan_pearson')
    print(overall_corr, overall_corr2)


    epilib.eric_plot_tri(D_sim[~nan_rows][:, ~nan_rows], D_no_nan, osp.join(max_ent_dir, 'dist_triu.png'),
                        vmaxp = np.nanmax(D),
                        title = f'Corr = {overall_corr}', cmap = BLUE_CMAP)

    meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(D)
    # print(meanDist_gt)
    np.save(osp.join(max_ent_dir, 'meanDist_gt.npy'), meanDist_gt)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_sim)
    plot_mean_dist(meanDist, max_ent_dir, 'dist_meanDist_log_ref.png',
                    None, True, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', None, 'Distance (nm)')
    plot_mean_dist(meanDist, max_ent_dir, 'dist_meanDist_ref.png',
                    None, False, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', None, 'Distance (nm)')
    plot_mean_dist(meanDist, max_ent_dir, 'dist_meanDist_ref.png',
                    None, False, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', None, 'Distance (nm)')


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
        plt.savefig(osp.join(max_ent_dir, 'dist_distance_pearson_log.png'))
    else:
        plt.savefig(osp.join(max_ent_dir, 'dist_distance_pearson.png'))
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

def get_dirs(dir, GNN_ID, b, phi, v, ar, k_angle=0, theta=180):
    if phi is None:
        assert v is not None
        max_ent_dir = osp.join(dir, f'optimize_grid_b_{b}_v_{v}')
        gnn_dir = osp.join(dir, f'optimize_grid_b_{b}_v_{v}')
    else:
        max_ent_dir = osp.join(dir, f'optimize_grid_b_{b}_phi_{phi}')
        gnn_dir = osp.join(dir, f'optimize_grid_b_{b}_phi_{phi}')

    if ar != 1.0:
        max_ent_dir += f'_spheroid_{ar}'
        gnn_dir += f'_spheroid_{ar}'
    if k_angle != 0:
        max_ent_dir += f'_angle_{k_angle}'
        gnn_dir += f'_angle_{k_angle}'
        if theta != 180:
            max_ent_dir += f'_theta0_{theta}'
            gnn_dir += f'_theta0_{theta}'
    max_ent_dir += '-max_ent10'
    gnn_dir += f'-GNN{GNN_ID}'
    if GNN_ID is None:
        gnn_dir = None
    return max_ent_dir, gnn_dir

def compare_pcs(sample, GNN_ID, b, phi, ar):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID, b=b, phi=phi, ar=ar)
    nan_rows = np.isnan(D[0])

    # compare PCs
    V_D, D_oe = get_pcs(D, nan_rows, return_input=True)
    V_D_pca, D_pca_oe = get_pcs(D_pca, nan_rows, return_input=True)
    V_D_gnn = get_pcs(D_gnn, nan_rows)

    # compare oe
    arr = np.array([D_oe, D_pca_oe])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    fig, (axcb12, ax1, ax2, ax3, axcb3) = plt.subplots(1, 5,
                                gridspec_kw={'width_ratios':[0.08, 1,1,1,0.08]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    for D_i, label, ax in zip([D_oe, D_pca_oe],
                                ['D', 'D_pca'],
                                [ax1, ax2]):
        s = sns.heatmap(D_i, linewidth = 0, vmin = vmin, vmax = vmax,
                        cmap = RED_BLUE_CMAP,
                        ax = ax, cbar_ax = axcb12)
        s.set_title(label, fontsize = 16)
        s.set_xticks([])
        s.set_yticks([])

    diff = D_oe - D_pca_oe
    vmin = np.nanpercentile(diff, 1)
    vmax = np.nanpercentile(diff, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    s = sns.heatmap(diff, linewidth = 0, vmin = vmin, vmax = vmax,
                    cmap = RED_BLUE_CMAP,
                    ax = ax3, cbar_ax = axcb3)
    s.set_title('Difference', fontsize = 16)
    s.set_xticks([])
    s.set_yticks([])


    axcb12.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'D_oe_vs_D_pca_oe.png'))
    plt.close()


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
        s = sns.heatmap(D_i, linewidth = 0, vmin = 0, vmax = np.nanmean(D_i),
                        cmap = BLUE_CMAP, ax = ax[i], cbar = False)

        corr = pearson_round(D[~nan_rows][:, ~nan_rows][triu_ind], D_i[triu_ind],
                            stat = 'nan_pearson')
        title = (f'{label}'
                f'\nCorr={np.round(corr, 3)}')
        s.set_title(title, fontsize = 16)

        if i > 0:
            s.set_yticks([])


    plt.tight_layout()
    plt.savefig(osp.join(dir, 'D_PCA_vs_GNN.png'))
    plt.close()

def compare_rg(sample, GNN_ID, b, phi, v, ar):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    result = load_import_log(dir)
    resolution = result['resolution']
    chrom = int(result['chrom'])
    odir = '/home/erschultz/TICG-chromatin/figures/distances'

    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz = np.load(xyz_file)

    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi, v, ar)
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
        if xyz_gnn is not None:
            xyz_gnn_size = xyz_gnn[:, left:right, :]
            gnn_rgs[i] = calculate_rg(xyz_gnn_size)

        ref_rgs[i] = calculate_rg(xyz_size, verbose = False)
        max_ent_rgs[i] = calculate_rg(xyz_max_ent_size)

    print(ref_rgs[:, 0])
    plt.errorbar(log_labels, ref_rgs[:, 0], ref_rgs[:, 1], color = 'k',
                label = 'Experiment')
    plt.errorbar(log_labels, max_ent_rgs[:, 0], max_ent_rgs[:, 1], color = 'b',
                label = 'Max Ent')
    if xyz_gnn is not None:
        plt.errorbar(log_labels, gnn_rgs[:, 0], gnn_rgs[:, 1], color = 'r',
                    label = 'GNN')

    X = np.linspace(log_labels[0], log_labels[-1], 100)
    Y = np.power(X, 1/3)
    Y = Y * ref_rgs[0, 0] / np.min(Y) * 1.05
    plt.plot(X, Y, label = '1/3', ls='dotted', color = 'gray')

    Y = np.power(X, 1/4)
    Y = Y * ref_rgs[0, 0] / np.min(Y) * 0.65
    plt.plot(X, Y, label = '1/4', ls='dashed', color = 'gray')

    plt.ylabel('Radius of Gyration', fontsize=16)
    plt.xlabel('Domain Size (bp)', fontsize=16)
    plt.legend(fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'rg_comparison.png'))

def compare_rg_pos(sample, GNN_ID, b, phi, v, ar):
    odir = '/home/erschultz/TICG-chromatin/figures/distances'

    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    result = load_import_log(dir)
    start = result['start_mb']
    end = result['end_mb']
    chrom = int(result['chrom'])
    resolution = result['resolution']
    resolution_mb = result['resolution_mb']

    # get experiment
    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz = np.load(xyz_file)

    # get max ent and GNN
    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi, v, ar)
    final_dir = get_final_max_ent_folder(max_ent_dir)
    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_max_ent = xyz_load(file, multiple_timesteps = True)
    _, m, _ = xyz_max_ent.shape

    # set up xlabels
    all_labels_float = np.linspace(start, end, m)
    all_labels_int = np.round(all_labels_float, 0).astype(int)
    genome_ticks = [0, m//3, 2*m//3, m-1]
    genome_labels = [f'{all_labels_int[i]}' for i in genome_ticks]
    print(genome_labels)

    if gnn_dir is not None and osp.exists(gnn_dir):
        file = osp.join(gnn_dir, 'production_out/output.xyz')
        xyz_gnn = xyz_load(file, multiple_timesteps = True)
    else:
        xyz_gnn = None

    window_sizes = [8, 16, 32]
    for i, size in enumerate(window_sizes):
        size_mb = size * resolution / 1000 / 1000
        print(size_mb, 'mb')
        length = int(m/size)
        X = np.linspace(0, m-size, length).astype(int)
        X += int(size/2) # center on interval
        # X_genome_labels = [all_labels_float[i] for i in ticks]

        for xyz_i, label in zip([xyz, xyz_max_ent, xyz_gnn],
                                ['Experiment', 'Max Ent', 'GNN']):
            if label == 'Experiment':
                continue
            mean_arr = np.zeros(length)
            std_arr = np.zeros(length)
            if xyz_i is None:
                print(f'{label} is None')
                continue

            left = 0; right = left + size; i=0
            while right <= m:
                xyz_i_size = xyz_i[:, left:right, :]
                mean, std = calculate_rg(xyz_i_size)
                mean_arr[i] = mean
                std_arr[i] = std

                left += size
                right += size
                i += 1

            plt.plot(X, mean_arr, label = label)

        plt.title(f'b={b}, phi={phi}, ar={ar}')
        plt.xticks(genome_ticks, labels = genome_labels)
        plt.ylabel('Radius of Gyration', fontsize=16)
        plt.xlabel('Genomic Position (Mp)', fontsize=16)
        plt.legend(fontsize=16)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'rg_comparison_{size}.png'))
        plt.close()

def compare_rg_matrix(sample, GNN_ID, b, phi, v, ar):
    odir = '/home/erschultz/TICG-chromatin/figures/distances'
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    result = load_import_log(dir)
    start_mb = result['start_mb']
    end_mb = result['end_mb']
    start = result['start']
    end = result['end']
    chrom = int(result['chrom'])
    resolution = result['resolution']
    resolution_mb = result['resolution_mb']


    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi, v, ar)
    final_dir = get_final_max_ent_folder(max_ent_dir)
    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_max_ent = xyz_load(file, multiple_timesteps = True)
    _, m, _ = xyz_max_ent.shape

    if gnn_dir is not None and osp.exists(gnn_dir):
        file = osp.join(gnn_dir, 'production_out/output.xyz')
        xyz_gnn = xyz_load(file, multiple_timesteps = True)
    else:
        xyz_gnn = None

    # get experiment
    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz = np.load(xyz_file)
    print('Experimental xyz:', xyz.shape)
    # crop
    data_dir = os.sep.join(dir.split(os.sep)[:-2])
    D = np.load(osp.join(data_dir, f'dist_mean_chr{chrom}.npy'))
    with open(osp.join(data_dir, f'coords_chr{chrom}.json')) as f:
        coords_dict = json.load(f)
    i = coords_dict[f'chr{chrom}:{start}-{start+resolution}']
    xyz = xyz[:, i:i+m, :]

    # set up xlabels
    all_labels_float = np.linspace(start_mb, end_mb, m)
    all_labels_int = np.round(all_labels_float, 0).astype(int)
    genome_ticks = [0, m-1]
    genome_labels = [f'{all_labels_int[i]}' for i in genome_ticks]
    print(genome_labels)
    rg_list = [None]*3
    files = ['rg_exp.npy', 'rg_me.npy', 'rg_gnn.npy']
    for i, (xyz_i, file_i, label) in enumerate(zip([xyz, xyz_max_ent, xyz_gnn],
                                            files,
                                            ['Experiment', 'Max Ent', 'GNN'])):
        if xyz_i is None:
            continue
        file_i = osp.join(dir, file_i)
        if osp.exists(file_i):
            rg_i = np.load(file_i)
            rg_list[i] = rg_i
        else:
            rg_list[i] = calculate_rg_matrix(xyz_i[:100], file_i, True)
    print('---'*9)
    tick_fontsize=18
    letter_fontsize=26
    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(6.15)
    fig.set_figwidth(12)

    indu = np.triu_indices(m)
    indl = np.tril_indices(m)
    composites = []
    for rg_i in rg_list[1:]:
        # make composite contact map
        composite = np.zeros((m, m))
        composite[indu] = rg_i[indu]
        composite[indl] = rg_list[0][indl]
        composites.append(composite)

    # plot rg matrices
    arr = np.array(composites)
    vmax = np.nanmean(arr)
    data = zip(composites, ['GNN', 'Max Ent'])
    for i, (composite, label) in enumerate(data):
        ax = axes[i]
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = BLUE_RED_CMAP,
                        ax = ax, cbar = True)
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*m, 0.01*m, label, fontsize=letter_fontsize, ha='right', va='top',
                weight='bold')
        ax.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold')
        # s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
        # s.set_yticks(genome_ticks, labels = genome_labels)
        s.set_xticks([])
        s.set_yticks([])
        s.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    for n, ax in enumerate(axes):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.savefig(osp.join(odir, f'rg_comparison_matrix.png'))
    plt.close()


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
    with open() as f:
        coords_dict = json.load(f)

    D = crop_hg38(D, f'chr{chrom}:{start}-{start+resolution}', len(y), coords_dict)

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

def compare_diagonal(sample, GNN_ID, b, phi, v, ar):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}'
    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, phi)
    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID, b=b, phi=phi, v=v, ar=ar)
    nan_rows = np.isnan(D[0])
    D_no_nan = D[~nan_rows][:, ~nan_rows] # ignore nan_rows
    alpha_pca = min_MSE(D_no_nan, D_pca[~nan_rows][:, ~nan_rows])
    alpha_gnn = min_MSE(D_no_nan, D_gnn[~nan_rows][:, ~nan_rows])
    # alpha_pca = 1; alpha_gnn = 1
    D_pca = D_pca * alpha_pca
    D_gnn = D_gnn * alpha_gnn
    # np.save(osp.join(dir, 'D.npy'), D)
    # np.save(osp.join(dir, 'D_gnn.npy'), D_gnn)
    # np.save(osp.join(dir, 'D_pca.npy'), D_pca)

    m = len(D_pca)

    # process experimental distance map
    nan_rows = np.isnan(D[0])
    D_no_nan = D[~nan_rows][:, ~nan_rows]

    plot_diagonal(D_no_nan, D_pca[~nan_rows][:, ~nan_rows], osp.join(dir, 'diagonal_pca.png'))
    plot_diagonal(D_no_nan, D_gnn[~nan_rows][:, ~nan_rows], osp.join(dir, 'diagonal_gnn.png'))

def compare_dist_ij(sample, GNN_ID, b, v, ar):
    dir = f'/home/erschultz/Su2020/samples/sample{sample}_rescale1'
    odir = '/home/erschultz/TICG-chromatin/figures/distances'

    D, D_gnn, D_pca = load_exp_gnn_pca(dir, GNN_ID, b=b, phi=None, v=v, ar=ar)
    nan_rows = np.isnan(D[0])
    m = len(D_pca)
    D_no_nan = D[~nan_rows][:, ~nan_rows] # ignore nan_rows
    D_pca_no_nan = D_pca[~nan_rows][:, ~nan_rows]

    # observed / expected
    D_no_nan = epilib.get_oe(D_no_nan)
    D_pca_no_nan = epilib.get_oe(D_pca_no_nan)

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

    x = D_no_nan.flatten()[::10]
    y = D_pca_no_nan.flatten()[::10]
    # Calculate the point density
    xy = np.vstack([x,y])
    print(xy.shape)
    z = gaussian_kde(xy)(xy)
    print('fitted kde')
    plt.scatter(x, y, c=z, s=10)
    plt.axline((0,0), slope=1, color = 'k')
    plt.xlabel(r'$D_{ij}$', fontsize=16)
    plt.ylabel(r'$D^{PCA}_{ij}$', fontsize=16)
    plt.xlim(0.5, None)
    plt.ylim(0.5, None)
    plt.tight_layout()
    # plt.show()
    plt.savefig(osp.join(odir, 'D_oe_ij.png'))
    plt.close()

def compare_bonded_p_s():
    data_dir = '/home/erschultz/Su2020/samples/sample1013'
    def load_data(dir):
        y = np.load(osp.join(dir, 'y.npy')).astype(float)
        y /= np.mean(y.diagonal())
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
        return meanDist

    meanDist = load_data(data_dir)
    plt.plot(meanDist, color = 'k', label = 'Experiment')


    for b, ls in zip([261, 140], ['-', '--']):
        for phi in [0.005, 0.01, 0.03]:
            for ar in [1.0]:
                if ar == 1.0:
                    grid_dir = osp.join(data_dir, f'optimize_grid_b_{b}_phi_{phi}')
                    label = f'b_{b}_phi_{phi}'
                else:
                    grid_dir = osp.join(data_dir, f'optimize_grid_b_{b}_phi_{phi}_spheroid_{ar}')
                    label = f'b_{b}_phi_{phi}_ar_{ar}'
                if not osp.exists(grid_dir):
                    print(f'Warning {grid_dir} does not exist')
                    continue
                meanDist = load_data(grid_dir)
                plt.plot(meanDist, label = label, ls=ls)

    plt.ylabel('Bonded Contact Probability')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

def compare_bonded():
    data_dir = '/home/erschultz/Su2020/samples/sample1013'
    m=512
    result = load_import_log(data_dir)
    start = result['start']
    resolution = result['resolution']
    chrom = int(result['chrom'])
    genome = result['genome']

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
        with open(osp.join(data_dir, f'coords_chr{chom}.json')) as f:
            coords_dict = json.load(f)

        D = crop_hg38(D, f'chr{chrom}:{start}-{start+resolution}', m, coords_dict)

    log_labels = np.linspace(0, resolution*(m-1), m)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(D, 'freq')
    nan_rows = np.isnan(meanDist)
    plt.plot(log_labels[~nan_rows], meanDist[~nan_rows], color = 'k', label = 'Experiment')

    def load_data(b, phi, ar=1.0, max_ent=False):
        if ar == 1.0:
            max_ent_dir = osp.join(data_dir, f'optimize_grid_b_{b}_phi_{phi}')
            label = f'b_{b}_phi_{phi}'
        else:
            max_ent_dir = osp.join(data_dir, f'optimize_grid_b_{b}_phi_{phi}_spheroid_{ar}')
            label = f'b_{b}_phi_{phi}_ar_{ar}'
        if max_ent:
            max_ent_dir, _ = get_dirs(data_dir, None, b, phi, ar)
            label += '_maxent'
        if osp.exists(max_ent_dir + '_run_longer'):
            max_ent_dir += '_run_longer'
        if osp.exists(max_ent_dir):
            D_pca, _ = sim_xyz_to_dist(max_ent_dir, True)
            assert m == len(D_pca)
        else:
            print(f'{max_ent_dir} does not exist')
            return None, None
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_pca, 'freq')
        return meanDist, label

    b=261
    for max_ent in [False, True]:
        for b, ls in zip([261, 140], ['-', '--']):
            for phi in [0.01]:
                for ar in [1.0]:
                    meanDist, label = load_data(b, phi, ar, max_ent)
                    if meanDist is None:
                        continue
                    plt.plot(log_labels, meanDist, label = label, ls=ls)

    plt.xscale('log')
    plt.ylabel('Spatial Distance (nm)', fontsize=16)
    plt.xlabel('Genomic Distance (bp)', fontsize=16)
    plt.legend()
    plt.show()

def jsd(sample, GNN_ID, b, v, ar):
    odir = '/home/erschultz/TICG-chromatin/figures/distances'
    dir = f'/home/erschultz/Su2020/samples/sample{sample}_rescale1'
    result = load_import_log(dir)
    start_mb = result['start_mb']
    end_mb = result['end_mb']
    start = result['start']
    end = result['end']
    chrom = int(result['chrom'])
    resolution = result['resolution']
    resolution_mb = result['resolution_mb']


    max_ent_dir, gnn_dir = get_dirs(dir, GNN_ID, b, None, v, ar)
    final_dir = get_final_max_ent_folder(max_ent_dir)
    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_max_ent = xyz_load(file, multiple_timesteps = True)
    _, m, _ = xyz_max_ent.shape

    if gnn_dir is not None and osp.exists(gnn_dir):
        file = osp.join(gnn_dir, 'production_out/output.xyz')
        print(file)
        xyz_gnn = xyz_load(file, multiple_timesteps = True)
    else:
        xyz_gnn = None

    # get experiment
    if chrom == 21:
        exp_dir = '/home/erschultz/Su2020/samples/sample1'
    elif chrom == 2:
        exp_dir = '/home/erschultz/Su2020/samples/sample10'
    xyz_file = osp.join(exp_dir, 'xyz.npy')
    xyz = np.load(xyz_file)
    print('Experimental xyz:', xyz.shape)

    # crop coordinates
    data_dir = os.sep.join(dir.split(os.sep)[:-2])
    with open(osp.join(data_dir, f'coords_chr{chrom}.json')) as f:
        coords_dict = json.load(f)
    i = coords_dict[f'chr{chrom}:{start}-{start+resolution}']
    xyz = xyz[:, i:i+m, :]

    # crop samples
    s_crop = 500
    D = xyz_to_distance(xyz[:s_crop], False)
    min_d = 0
    max_d = 3000

    # set up xlabels
    all_labels_float = np.linspace(start_mb, end_mb, m)
    all_labels_int = np.round(all_labels_float, 0).astype(int)
    genome_ticks = [0, m-1]
    genome_labels = [f'{all_labels_int[i]}' for i in genome_ticks]
    print(genome_labels)

    for i, (xyz_i, label) in enumerate(zip([xyz_gnn],
                                            ['GNN'])):
        if xyz_i is None:
            continue
        D_i = xyz_to_distance(xyz_i[:s_crop], False)

        JSD = np.zeros((m,m))
        bins=40
        for j in range(m):
            if j % 50 == 0:
                print(j)
            for k in range(j, m):
                left, bins = np.histogram(D[:,j,k], bins, (min_d, max_d))
                left = left.astype(float) / np.sum(left)
                right, _ = np.histogram(D_i[:,j,k], bins, (min_d, max_d))
                right = right.astype(float) / np.sum(right)
                jsd = jensenshannon(left, right)**2
                JSD[j,k] = jsd
                JSD[k,j] = jsd
                if (j - k) % 50 == 0 and j % 50 == 0:
                    print('\t\t', j, k)
                    # plt.hist(D[:,j,k], bins, (min_d, max_d), label='ref',
                    #         alpha=0.5, weights = np.ones_like(D[:,j,k]) / len(D[:,j,k]))
                    # plt.hist(D_i[:,j,k], bins, (min_d, max_d), label='sim',
                    #         alpha=0.5, weights = np.ones_like(D[:,j,k]) / len(D[:,j,k]))
                    plt.stairs(left, bins, label='ref', alpha=0.5, fill=True)
                    plt.stairs(right, bins, label='sim', alpha=0.5, fill=True)
                    plt.legend(loc='upper right')
                    plt.title(f'JSD={np.round(jsd, 6)}')
                    plt.savefig(osp.join(odir, f'temp/hist_{j}_{k}.png'))
                    plt.close()
        print(JSD, np.nanmean(JSD))


    print('---'*9)
    tick_fontsize=18
    letter_fontsize=26
    plot_matrix(JSD, osp.join(odir, 'jsd_matrix'),
                cmap=plt.get_cmap("plasma"), vmax=0.4)
    # fig, axes = plt.subplots(1, 2)
    # fig.set_figheight(6.15)
    # fig.set_figwidth(12)
    # plt.tight_layout()
    # plt.savefig(osp.join(odir, f'jsd_matrix.png'))
    # plt.close()


if __name__ == '__main__':
    dir = '/home/erschultz/Su2020'
    # tsv_to_npy()
    # test_pcs()
    # load_exp_gnn_pca(osp.join(dir, 'samples/sample1002'))
    xyz_to_dist2()
    # xyz_to_xyz()
    # compare_D_to_sim_D(10)
    # compare_diagonal(1013, 434)
    # sim_xyz_to_dist(osp.join(dir, 'samples/sample1011/optimize_grid_b_140_phi_0.03-GNN403'),
    #                 False)
    # find_volume()
    # compare_bonded()
    # compare_pcs(1013, None, b=180, phi=0.01, ar=2.0)
    # compare_d_maps(1003, None)
    # compare_dist_distribution_a_b()
    # compare_dist_distribution_plaid(1013, None, 261, 0.01)
    # compare_rg('1004_rescale1', None, b=200, phi=None, v=8, ar=1.5)
    # compare_rg_pos('1004_rescale1', 690, b=200, phi=None, v=8, ar=1.5)
    # compare_rg_matrix('1004_rescale1', 631, b=200, phi=None, v=8, ar=1.5)
    # compare_dist_ij(1004, None, b=200, v=8, ar=1.5)
    # jsd(1004, 631, b=200, v=8, ar=1.5)
    # compare_scaling(1002, None, 261, 0.006)
