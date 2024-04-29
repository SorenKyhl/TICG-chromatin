import csv
import os
import os.path as osp
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.utils import load_json, print_time
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import nan_euclidean_distances


def lammps_load(filepath, save = False, N_min = None, N_max = None, down_sampling = 1):
    xyz_npy_file = osp.join(osp.split(filepath)[0], 'xyz.npy')
    x_npy_file = osp.join(osp.split(filepath)[0], 'x.npy')
    t0 = time.time()
    if osp.exists(xyz_npy_file):
        xyz = np.load(xyz_npy_file)
    else:
        xyz = []
        with open(filepath, 'r') as f:
            line = 'null'
            while line != '':
                line = f.readline().strip()
                if line == 'ITEM: NUMBER OF ATOMS':
                    N = int(f.readline().strip())
                    xyz_timestep = np.empty((N, 3))

                if line == 'ITEM: ATOMS id type xu yu zu':
                    line = f.readline().strip().split(' ')
                    while line[0].isnumeric():
                        i = int(line[0]) - 1
                        xyz_timestep[i, :] = [float(j) for j in line[2:5]]
                        if i == N-1:
                            xyz.append(xyz_timestep)
                        line = f.readline().strip().split(' ')
        xyz = np.array(xyz)
        if save:
            np.save(xyz_npy_file, xyz)

    if osp.exists(x_npy_file):
        x = np.load(x_npy_file)
    else:
        x = []
        with open(filepath, 'r') as f:
            keep_reading = True
            while keep_reading:
                line = f.readline().strip()
                if line == 'ITEM: ATOMS id type xu yu zu':
                    keep_reading = False
                    line = f.readline().strip().split(' ')
                    while line[0].isnumeric():
                        i = int(line[0]) - 1
                        x.append(int(line[1])-1)
                        line = f.readline().strip().split(' ')
        N = len(x)
        x_arr = np.zeros((N, np.max(x)+1))
        x_arr[np.arange(N), x] = 1
        if save:
            np.save(x_npy_file, x_arr)

    if N_min is None:
        N_min = 0
    if N_max is None:
        N_max = len(xyz)
    xyz = xyz[N_min:N_max:down_sampling]
    tf = time.time()
    print(f'Loaded xyz with shape {xyz.shape}')
    print_time(t0, tf, 'xyz load')
    return xyz, x_arr

def xyz_load(xyz_filepath, delim = '\t', multiple_timesteps = True, save = False,
            N_min = 1, N_max = None, down_sampling = 1,
            verbose = True, throw_exception=True):
    t0 = time.time()
    xyz_npy_file = osp.join(osp.split(xyz_filepath)[0], 'xyz.npy')
    if osp.exists(xyz_npy_file):
        xyz = np.load(xyz_npy_file)
    else:
        xyz = []
        with open(xyz_filepath, 'r') as f:
            N = int(f.readline())
            reader = csv.reader(f, delimiter = delim)
            xyz_timestep = np.empty((N, 3))
            for line in reader:
                if N_max is not None and len(xyz) == N_max:
                    break
                if len(line) > 1:
                    i = int(line[0])
                    try:
                        xyz_timestep[i, :] = [float(j) for j in line[1:4]]
                    except:
                        print(line)
                        if throw_exception:
                            raise
                        else:
                            return None
                    if i == N-1:
                        xyz.append(xyz_timestep)
                        xyz_timestep=np.empty((N, 3))

        xyz = np.array(xyz)
        if save:
            np.save(xyz_npy_file, xyz)
    if N_min is None:
        N_min = 0
    if N_max is None:
        N_max = len(xyz)
    if not multiple_timesteps:
        xyz = xyz[N_min]
    else:
        xyz = xyz[N_min:N_max:down_sampling]

    tf = time.time()
    if verbose:
        print(f'Loaded xyz with shape {xyz.shape}')
        print(f'time: {np.round(tf - t0, 3)} s')
    return xyz

def xyz_load_cores(filepath, save = False, verbose = True):
    t0 = time.time()
    xyz_npy_file = osp.join(filepath, 'xyz.npy')
    if osp.exists(xyz_npy_file):
        xyz = np.load(xyz_npy_file)
    else:
        xyz = []
        core_files = [f for f in os.listdir(filepath) if f.startswith('core')]
        for core in core_files:
            xyz_filepath = osp.join(filepath, core, 'output.xyz')
            if verbose:
                print(xyz_filepath)
            xyz_core = xyz_load(xyz_filepath, verbose = verbose, save = True,
                                throw_exception = False)
            if xyz_core is not None:
                xyz.append(xyz_core)

        if verbose:
            print(f'Using {len(xyz)} cores, each core is shape {xyz[0].shape}')

        xyz = np.concatenate(xyz)
        if save:
            np.save(xyz_npy_file, xyz)

    tf = time.time()
    if verbose:
        print(f'Loaded xyz with shape {xyz.shape}')
        print(f'time: {np.round(tf - t0, 3)} s')

    return xyz


def xyz_write(xyz, outfile, writestyle, comment = '', x = None):
    '''
    Write the coordinates of all particle to a file in .xyz format.
    Inputs:
        xyz: shape (T, N, 3) or (N, 3) array of all particle positions (angstroms)
        outfile: name of file
        writestyle: 'w' (write) or 'a' (append)
        x: additional columns to include
    '''
    if writestyle == 'w' and osp.exists(outfile):
        os.remove(outfile)
    if len(xyz.shape) == 3:
        N, m, _ = xyz.shape
        for i in range(N):
            xyz_write(xyz[i, :, :], outfile, 'a', comment = comment, x = x)
    else:
        m, _ = xyz.shape
        if x is not None:
            assert len(x) == m, f'{len(x)} != {m}'
            _, k = x.shape

        with open(outfile, writestyle) as f:
            f.write(f'{m}\n{comment}\n')
            for i in range(m):
                row = f'{i} {xyz[i,0]} {xyz[i,1]} {xyz[i,2]}'
                if x is not None:
                    for j in range(k):
                        row += f' {x[i,j]}'
                f.write(row + '\n')

def xyz_to_contact_grid(xyz, grid_size, sparse_format = False, dtype = np.int32,
                        verbose = False):
    '''
    Converts xyz to contact map via grid.

    Inputs:
        xyz: np array of shape N, m, 3 (N is optional)
        grid_size: size of grid (nm)
    '''
    if len(xyz.shape) == 3:
        N, m, _ = xyz.shape
    else:
        N = 1
        m, d = xyz.shape
        xyz = xyz.reshape(1, m, d)

    contact_map = np.zeros((m, m)).astype(dtype)
    t0 = time.time()
    for n in range(N):
        if verbose:
            prcnt_done = n/N * 100
            t = time.time() - t0
            if prcnt_done % 5 == 0:
                print(f'{prcnt_done}%')
        # use dictionary to find contacts
        grid_dict = defaultdict(list) # grid (x, y, z) : bead id list
        for i in range(m):
            grid_i = tuple([d // grid_size for d in xyz[n, i, :3]])
            grid_dict[grid_i].append(i)

        for bead_list in grid_dict.values():
            for i in bead_list:
                for j in bead_list:
                    contact_map[i,j] += 1

    if sparse_format:
        return csr_array(contact_map)

    return contact_map

def xyz_to_contact_distance(xyz, cutoff_distance, dtype = np.int32, verbose = False):
    '''
    Converts xyz to contact map via grid
    '''
    if len(xyz.shape) == 3:
        N = xyz.shape[0]
        m = xyz.shape[1]
    else:
        N = 1
        m = xyz.shape[0]
        xyz = xyz.reshape(-1, m, 3)

    t0 = time.time()
    D = xyz_to_distance(xyz, verbose)
    contact_map = D <= cutoff_distance
    contact_map = np.sum(contact_map, axis = 0).astype(dtype)

    return contact_map


def xyz_to_distance(xyz, verbose = False):
    N, m, _ = xyz.shape
    D = np.zeros((N, m, m), dtype = np.float32)
    for i in range(N):
        if verbose and i % 100 == 0:
            print(i)
        D_i = nan_euclidean_distances(xyz[i])
        D[i] = D_i

    return D

def xyz_to_angles(xyz, verbose = False):
    '''Compute N x m-1 array of bond angles.'''
    N, m, _ = xyz.shape
    angles = np.zeros((N, m-2), dtype=np.float32)
    for i in range(N):
        if verbose:
            print(i)
        angles_temp = []
        for j in range(1, m-1):
            A = xyz[i, j-1, :]
            C = xyz[i, j, :]
            B = xyz[i, j+1, :]
            # use law of cosines
            # https://www.mathsisfun.com/algebra/trig-cosine-law.html
            a = np.linalg.norm(B-C)
            b = np.linalg.norm(A-C)
            c = np.linalg.norm(A-B)
            num = a**2 + b**2 - c**2
            denom = 2 * a * b
            angle = np.rad2deg(np.arccos(num/denom))
            angles_temp.append(angle)
        angles[i] = angles_temp
    return angles

def calculate_rg_old(xyz, verbose=False):
    if len(xyz.shape) == 2:
        xyz.reshape(1, -1, 3)

    rgs = np.zeros(len(xyz))
    for i, xyz_i in enumerate(xyz):
        center = np.nanmean(xyz_i, axis = 0)
        delta = xyz_i - center
        rg = np.sqrt(np.nanmean(delta**2))
        rgs[i] = rg

    rg_mean = np.nanmean(rgs)
    rg_std = np.nanstd(rgs)
    result = (rg_mean, rg_std)
    if verbose:
        print('rgs', rgs)
        print('result', result)
    return result

def calculate_rg(xyz, verbose=False, full_distribution=False):
    if len(xyz.shape) == 2:
        xyz.reshape(1, -1, 3)
    N, m, _ = xyz.shape

    # this is equivalent
    # rgs = np.zeros(len(xyz))
    # for i, xyz_i in enumerate(xyz):
    #     mean_i = np.nanmean(xyz_i, axis = 0)
    #     center_i = xyz_i - mean_i
    #     delta_i_2 = np.nansum(center_i**2, 1)
    #     rg = np.sqrt(np.nanmean(delta_i_2))
    #     rgs[i] = rg

    xyz_mean = np.nanmean(xyz, axis = 1)
    xyz = np.transpose(xyz, (1, 0, 2)) # transpose so broadcasting works (m, N, 3)
    center = xyz - xyz_mean
    delta_2 = np.nansum(center**2, 2) # sum over spatial dimension
    rgs = np.sqrt(np.nanmean(delta_2, axis=0)) # mean over particles

    if verbose:
        print('rgs', rgs)

    if full_distribution:
        return rgs
    else:
        rg_mean = np.nanmean(rgs)
        rg_std = np.nanstd(rgs)
        result = (rg_mean, rg_std)
        if verbose:
            print('result', result)
        return result

def calculate_rg_matrix(xyz, ofile=None, verbose=False):
    N, m, _ = xyz.shape
    rg_arr = np.zeros((m,m))
    for i in range(m):
        if verbose and i % 10 == 0:
            print(i)
        for j in range(i+2, m):
            mean, std = calculate_rg(xyz[:, i:j, :])
            rg_arr[i, j] = mean

    # make symmetric
    rg_arr = rg_arr + rg_arr.T

    if ofile is not None:
        np.save(ofile, rg_arr)

    return rg_arr

def time_contact_distance():
    N=2
    m=4
    xyz = np.random.rand(N*m*3).reshape(N, m, 3)
    xyz_to_contact_distance(xyz, 0.5, verbose=True)

def compare_python_to_cpp():
    dir = osp.join('/home/erschultz/dataset_bonded/boundary_spheroid_2.0',
                    'bond_type_gaussian/m_512/bond_length_180/v_8/angle_0')
    y_cpp = None
    xyz = xyz_load(osp.join(dir, 'production_out/output.xyz'))
    dir = osp.join('/home/erschultz/dataset_06_29_23/samples/sample1/',
                    'optimize_distance_b_180_v_8_spheroid_1.5-max_ent10/iteration0')
    # y_cpp = np.load(osp.join(dir, 'y.npy'))
    config = load_json(osp.join(dir, 'config.json'))
    cutoff = config['distance_cutoff']
    print(cutoff)
    y_p = xyz_to_contact_distance(xyz, cutoff, verbose=True)

    for y, name in zip([y_p, y_cpp], ['python', 'cpp']):
        if y is None:
            continue
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label=name)

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Contact Probability', fontsize = 16)
    plt.xlabel('Polymer Distance (beads)', fontsize = 16)
    plt.legend()
    # plt.set_ylim(10**-5, None)
    plt.tight_layout()
    plt.show()

def test_calc_rg():
    N = 1
    m = 15
    np.random.seed(12)
    xyz = np.random.rand(N*m*3).reshape(N, m, 3)
    print('xyz', xyz.shape)
    t0 = time.time()
    mean, _ = calculate_rg(xyz)
    tf = time.time()
    print_time(t0, tf, 'calculate_rg')
    print(mean)

    # t0 = time.time()
    # mean, _ = calculate_rg_old(xyz)
    # tf = time.time()
    # print_time(t0, tf, 'calculate_rg_old')
    # print(mean)

    t0 = time.time()
    # rg_arr = calculate_rg_matrix(xyz[:100], verbose=True)
    # print(rg_arr)
    tf = time.time()
    print_time(t0, tf, 'calculate_rg_matrix')

def Q_similarity(D1, D2, delta, diff=False):
    """
    Calculate Q collective variable given distances matrices D1, D2.

    Inputs:
        D1, D2: distance matrices for two individual configurations
        delta: resolution length scale
        diff: True to return 1 - Q
    """
    assert len(D1) == len(D1)
    m = len(D1) # number of particles

    D_diff = D1-D2
    D_diff_squared = D_diff ** 2
    denom = 2 * delta**2
    values = np.exp(-D_diff_squared/denom)
    triu_ind = np.triu_indices(m, 1)
    Q = np.nanmean(values[triu_ind])

    if diff:
        return 1 - Q
    else:
        return Q

def test_Q():
    m = 10
    D1 = np.random.normal(size=(m,m))
    D1[0, 5] = np.nan
    D2 = np.random.normal(size=(m,m))

    # print(D1)
    Q = Q_similarity(D1, D2, delta = 165)
    print(Q)

def test_load_cores():
    dir = '/home/erschultz/soren_scaleup/20k-1k/final-20480'
    xyz = xyz_load_cores(dir)
    print(xyz.shape)

if __name__ == '__main__':
    # time_contact_distance()
    # test_calc_rg()
    # test_Q()
    test_load_cores()
    # compare_python_to_cpp()
