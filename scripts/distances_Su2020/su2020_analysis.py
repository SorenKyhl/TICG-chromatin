import csv
import json
import math
import os
import os.path as osp
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import seaborn as sns
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import (BLUE_CMAP, RED_BLUE_CMAP, plot_matrix,
                                        plot_mean_dist)
from pylib.utils.similarity_measures import SCC
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.argparse_utils import ArgparserConverter
from sequences_to_contact_maps.scripts.load_utils import (
    get_final_max_ent_folder, load_Y)
from sequences_to_contact_maps.scripts.utils import (calc_dist_strat_corr,
                                                     nan_pearsonr,
                                                     pearson_round)
from sequences_to_contact_maps.scripts.xyz_utils import (calculate_rg,
                                                         xyz_load,
                                                         xyz_to_distance)


# plotting functions
def rotate_bound(image, angle):
    def plot_diagonal(exp, sim):

        #This colors the image
        resized = exp

        # Rotate 45 degs
        resized = rotate_bound(resized,-45)


        fig, ax = plt.subplots(figsize=(6, 6),dpi=600)
        _pf = ax.imshow(resized, vmin=200, vmax=1000, cmap='seismic')

        plt.show()

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),cv2.INTER_NEAREST)

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
def crop_hg38(inp, start, m):
    '''
    Crop input distance map to m pixels starting from start.

    Inputs:
        inp (np array): distance map
        start (str): genomic coordinates
        m (int): desired len of output
    '''
    dir = '/home/erschultz/Su2020/samples/sample1'
    coords_file = osp.join(dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)

    i = coords_dict[start]
    print(i)
    return inp[i:i+m, i:i+m]

def hg38_to_hg19(coords):
    '''Convert hg38 coords to hg19.'''
    dir = '/home/erschultz/Su2020/samples/sample1'
    with open(osp.join(dir, 'hg38_positions.txt'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            if line.strip() == coords:
                break

    with open(osp.join(dir, 'hg19_positions.bed')) as f:
        return f.readlines()[i]

def min_MSE(D, D_sim):
    def metric(D, D_sim, alpha):
        return mean_squared_error(D, D_sim*alpha)

    popt, pcov = curve_fit(metric, D, D_sim, p0 = 1, maxfev = 2000)
    print(f'popt', popt)

    return popt

def get_pcs(input, nan_rows=None):
    if nan_rows is None:
        nan_rows = np.isnan(input[0])
    input = input[~nan_rows][:, ~nan_rows] # ignore nan_rows

    seqs = epilib.get_pcs(epilib.get_oe(input), 12,
                        normalize = False, align = True)
    return seqs.T # k x m


# analysis scripts
def find_hg38_positions():
    dir = '/home/erschultz/Su2020/samples/sample1'
    file = osp.join(dir, 'Hi-C_contacts_chromosome21.tsv') # hic file
    df = pd.read_csv(file, sep = '\t', index_col=0)
    lower_ind = 4
    upper_ind = 4 + 512

    # df.columns contains genomic coordinate windows
    cols = [i.split(':')[1].split('-') for i in df.columns]

    cols = [(int(i), int(j)) for i,j in cols]
    print(cols) # cols now contains tuples of (start, end) coordinate for each window

    # find contiguous region of 512 windows
    for i, (start, end) in enumerate(cols):
        if i <= lower_ind:
            prev_end = end
            continue

        if not start == prev_end:
            diff = start-prev_end
            print(f'{i} {prev_end} {start}, diff={diff}, {diff//50000}')
            upper_ind -= diff // 50000

        prev_end = end

    columns = df.columns[lower_ind:upper_ind+1]
    cols = cols[lower_ind:upper_ind+1]

    print((cols[-1][1] - cols[0][0]) / 50000)

    with open(osp.join(dir, 'hg38_positions.txt'), 'w') as f:
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

def compare_dist_distribution():
    '''Compare distributions of A-B distances between sim and experiment.'''
    def dist_distribution(xyz, a, b):
        # a and b are indices
        num_cells, num_coords, _ = xyz.shape
        dist = np.zeros(num_cells)
        for i in range(num_cells):
            dist[i] = np.linalg.norm(xyz[i, a, :] - xyz[i, b, :])
        return dist

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
    dist = dist_distribution(xyz, a, b)

    dir = '/home/erschultz/Su2020/samples/sample1002'
    max_ent_dir = osp.join(dir, 'optimize_grid_b_261_phi_0.006-max_ent10')
    final_dir = get_final_max_ent_folder(max_ent_dir)
    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_sim = xyz_load(file, multiple_timesteps = True)
    with open(osp.join(dir, 'import.log'), 'r') as f:
        line = f.readline().strip()
        while not line.startswith('start'):
            line = f.readline().strip()
        start = int(line.split('=')[1])
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

    dist_sim = dist_distribution(xyz_sim, a, b)


    print('exp', dist, dist.shape)
    print('sim', dist_sim, dist_sim.shape)

    bin_width = 50
    arr = dist[~np.isnan(dist)]
    plt.hist(arr, label = 'Exp', alpha = 0.5,
                weights = np.ones_like(arr) / len(arr),
                bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width))
    arr = dist_sim * 1.86
    plt.hist(arr, label = 'Sim', alpha = 0.5,
                weights = np.ones_like(arr) / len(arr),
                bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width))
    plt.xlabel('Distance between A and B', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.legend()
    plt.title(f'Distance between\n{coords_a} and {coords_b}')
    # plt.title(f'Distance between A and B')
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'distance_distribution.png'))

def xyz_to_dist():
    '''Convert experimental xyz to distance map.'''
    chr=2
    dir = '/home/erschultz/Su2020/samples/sample10'
    log_file = osp.join(dir, 'xyz_to_dist.log')
    log_file = open(log_file, 'w')
    file = osp.join(dir, f'chromosome{chr}.tsv')
    xyz_file = osp.join(dir, 'xyz.npy')
    coords_file = osp.join(dir, 'coords.json')
    D_file = osp.join(dir, 'dist_mean.npy')
    D_file2 = osp.join(dir, 'dist_median.npy')

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

    D = xyz_to_distance(xyz[:100], True)

    D_mean = np.nanmean(D, axis = 0)
    print(D_mean, file = log_file)
    print('mean', np.nanmean(np.diagonal(D_mean, 1)), file = log_file)
    np.save(D_file, D_mean)

    D_median = np.nanmedian(D, axis = 0)
    print(D_median, file = log_file)
    print('mean', np.nanmean(np.diagonal(D_median, 1)), file = log_file)
    print('median', np.nanmean(np.diagonal(D_median, 1)), file = log_file)

    np.save(D_file2, D_median)

    # create negative control
    D_file1 = osp.join(dir, 'dist_mean_first_half.npy')
    D_file2 = osp.join(dir, 'dist_mean_second_half.npy')
    D1 = D[:len(D)//2]
    D1_mean = np.nanmean(D1, axis = 0)
    np.save(D_file1, D1_mean)

    D2 = D[len(D)//2:]
    D2_mean = np.nanmean(D2, axis = 0)
    np.save(D_file2, D2_mean)

    log_file.close()

def sim_xyz_to_dist(dir, max_ent=True):
    if max_ent:
        final_dir = get_final_max_ent_folder(dir)
        file = osp.join(final_dir, 'production_out/output.xyz')
    else:
        file = osp.join(dir, 'production_out/output.xyz')
    D_file = osp.join(dir, 'dist_mean.npy')

    xyz = xyz_load(file, multiple_timesteps = True, N_min = 5)
    D = xyz_to_distance(xyz)

    D_mean = np.nanmean(D, axis = 0)
    # print(D_mean)
    # print('mean', np.nanmean(np.diagonal(D_mean, 1)))
    np.save(D_file, D_mean)

    return D_mean

def controls():
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
            ax[row,col].legend()

            col += 1
            if col > cols-1:
                col = 0
                row += 1
        plt.savefig(osp.join(max_ent_dir, 'pc_D_vs_Y.png'))

def compare_D_to_sim_D():
    dir = '/home/erschultz/Su2020/samples/sample1002'
    max_ent_dir = osp.join(dir, 'optimize_grid_b_261_phi_0.006-max_ent10')
    # max_ent_dir = '/home/erschultz/dataset_bond/samples/sample9'
    max_ent = True
    D_sim = sim_xyz_to_dist(max_ent_dir, max_ent)
    D_sim *= 1
    m = len(D_sim)

    # load config params
    if max_ent:
        with open(osp.join(max_ent_dir, 'resources/config.json'), 'r') as f:
            config = json.load(f)
            bl = config["bond_length"]
            delta = config["grid_size"]
            vb = config['beadvol']
    else:
        with open(osp.join(max_ent_dir, 'config.json'), 'r') as f:
            config = json.load(f)
            bl = config["bond_length"]
            delta = config["grid_size"]
            vb = config['beadvol']

    # process experimental distance map
    dir = '/home/erschultz/Su2020/samples/sample1'
    D_file = osp.join(dir, 'dist_mean.npy')
    D = np.load(D_file)
    nan_rows = np.isnan(D[0])
    D_no_nan = D[~nan_rows][:, ~nan_rows]
    plot_distance_map(D_no_nan, dir, 'exp_no_nan')
    D = crop_hg38(D, 'chr21:14000001-14050001', m)
    nan_rows = np.isnan(D[0])
    plot_distance_map(D_sim, max_ent_dir, 'sim')

    # min_MSE(D, D_sim)

    print(D.shape, D_sim.shape)
    triu_ind = np.triu_indices(len(D))
    overall_corr = pearson_round(D[triu_ind], D_sim[triu_ind], stat = 'nan_pearson')
    overall_corr2 = pearson_round(D, D_sim, stat = 'nan_pearson')
    print(overall_corr, overall_corr2)


    epilib.eric_plot_tri(D_sim, D, osp.join(max_ent_dir, 'dist_triu.png'), vmaxp = np.nanmax(D),
            title = f'Corr = {overall_corr}', cmap = BLUE_CMAP)

    title = f'b={bl}, gs={delta}, vb={vb}'
    meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(D)
    np.save(osp.join(max_ent_dir, 'meanDist_gt.npy'), meanDist_gt)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(D_sim)
    plot_mean_dist(meanDist, max_ent_dir, 'dist_meanDist_log_ref.png',
                    None, True, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', title, 'Distance (nm)')
    plot_mean_dist(meanDist, max_ent_dir, 'dist_meanDist_ref.png',
                    None, False, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', title, 'Distance (nm)')
    plot_mean_dist(meanDist, max_ent_dir, 'dist_meanDist_ref.png',
                    None, False, True, meanDist_gt, 'Exp', 'Sim',
                    'blue', title, 'Distance (nm)')


    # scc = SCC()
    # corr_scc = scc.scc(D, D_sim, var_stabilized = False)
    # corr_scc_var = scc.scc(D, D_sim, var_stabilized = True)
    avg_diag, corr_arr = calc_dist_strat_corr(D, D_sim, mode = 'nan_pearson',
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
        ax[row,col].legend()

        col += 1
        if col > cols-1:
            col = 0
            row += 1
    plt.savefig(osp.join(max_ent_dir, 'pc_D_vs_D_sim.png'))

def load_exp_gnn_pca(dir):
    max_ent_dir = osp.join(dir, 'optimize_grid_b_261_phi_0.006-max_ent10')
    D_pca = sim_xyz_to_dist(max_ent_dir, True)
    m = len(D_pca)
    gnn_dir = osp.join(dir, 'optimize_grid_b_140_phi_0.03-GNN403')
    D_gnn = sim_xyz_to_dist(gnn_dir, False)

    # process experimental distance map
    exp_dir = '/home/erschultz/Su2020/samples/sample1'
    D_file = osp.join(exp_dir, 'dist_mean.npy')
    D = np.load(D_file)
    D = crop_hg38(D, 'chr21:14000001-14050001', m)
    np.save(osp.join(dir, 'D_crop.npy'), D)

    return D, D_gnn, D_pca

def compare_pcs():
    dir = '/home/erschultz/Su2020/samples/sample1002'
    D, D_gnn, D_pca = load_exp_gnn_pca(dir)
    nan_rows = np.isnan(D[0])


    # compare PCs
    V_D = get_pcs(D, nan_rows)
    V_D_pca = get_pcs(D_pca, nan_rows)
    V_D_gnn = get_pcs(D_gnn, nan_rows)


    rows = 2; cols = 1
    row = 0; col = 0
    fig, ax = plt.subplots(rows, cols)
    fig.set_figheight(12)
    fig.set_figwidth(16)
    for i in range(rows*cols):
        ax[row].plot(V_D[i], label = 'Experiment', color = 'k')
        ax[row].plot(V_D_pca[i], label = 'Max. Ent.', color = 'blue')
        ax[row].plot(V_D_gnn[i], label = 'GNN', color = 'red')
        ax[row].set_title(f'PC {i+1}')
        if i == 0:
            ax[row].legend()

        col += 1
        if col > cols-1:
            col = 0
            row += 1
    plt.savefig(osp.join(dir, 'pc_D_vs_D_sim.png'))
    plt.close()

def compare_d_maps():
    dir = '/home/erschultz/Su2020/samples/sample1002'
    D, D_gnn, D_pca = load_exp_gnn_pca(dir)
    D_list = [D, D_pca, D_gnn]
    labels = ['Experiment', 'Max Ent', 'GNN']

    triu_ind = np.triu_indices(len(D))

    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,1]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    vmax = np.nanmean(D)
    for i, (D_i, label) in enumerate(zip(D_list, labels)):
        # if i == 2:
        #     s = sns.heatmap(D_i, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_CMAP,
        #                     ax = ax[i], cbar_ax = ax[3])
        # else:
        s = sns.heatmap(D_i, linewidth = 0, vmin = 0, vmax = np.nanmean(D_i), cmap = BLUE_CMAP,
                        ax = ax[i], cbar = False)

        if i == 0:
            s.set_title(label, fontsize = 16)
        else:
            corr = pearson_round(D[triu_ind], D_i[triu_ind], stat = 'nan_pearson')

            s.set_yticks([])
            title = (f'{label}'
                    f'\nCorr={np.round(corr, 3)}')
            s.set_title(title, fontsize = 16)



    plt.tight_layout()
    plt.savefig(osp.join(dir, 'D_PCA_vs_GNN.png'))
    plt.close()

def compare_rg():
    dir = '/home/erschultz/Su2020/samples/sample1'
    xyz_file = osp.join(dir, 'xyz.npy')
    xyz = np.load(xyz_file)
    coords_file = osp.join(dir, 'coords.json')
    with open(coords_file) as f:
        coords_dict = json.load(f)
    i = coords_dict['chr21:14000001-14050001']
    xyz = xyz[:, i:i+512, :]

    dir = '/home/erschultz/Su2020/samples/sample1002'
    max_ent_dir = osp.join(dir, 'optimize_grid_b_261_phi_0.006-max_ent10')
    final_dir = get_final_max_ent_folder(max_ent_dir)

    file = osp.join(final_dir, 'production_out/output.xyz')
    xyz_max_ent = xyz_load(file, multiple_timesteps = True)

    gnn_dir = osp.join(dir, 'optimize_grid_b_140_phi_0.03-GNN403')
    file = osp.join(gnn_dir, 'production_out/output.xyz')
    xyz_gnn = xyz_load(file, multiple_timesteps = True)

    sizes = [64, 128, 256, 512]
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

    plt.errorbar(sizes, ref_rgs[:, 0], ref_rgs[:, 1], color = 'k', label = 'Experiment')
    plt.errorbar(sizes, max_ent_rgs[:, 0], max_ent_rgs[:, 1], color = 'b', label = 'Max Ent')
    plt.errorbar(sizes, gnn_rgs[:, 0], gnn_rgs[:, 1], color = 'r', label = 'GNN')
    plt.ylabel('Radius of Gyration', fontsize=16)
    plt.xlabel('Domain Size (beads)', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'rg_comparison.png'))





if __name__ == '__main__':
    # load_exp_gnn_pca('/home/erschultz/Su2020/samples/sample1002')
    # find_hg38_positions()
    # xyz_to_dist()
    compare_D_to_sim_D()
    # find_volume()
    # compare_pcs()
    # compare_d_maps()
    # compare_dist_distribution()
    # compare_rg()
