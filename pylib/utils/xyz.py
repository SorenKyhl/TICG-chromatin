import csv
import json
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import nan_euclidean_distances


def xyz_load(xyz_filepath, delim = '\t', multiple_timesteps = True, save = False,
            N_min = 1, N_max = None, down_sampling = 1, verbose = True):
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
                if len(line) > 1:
                    i = int(line[0])
                    try:
                        xyz_timestep[i, :] = [float(j) for j in line[1:4]]
                    except:
                        print(line)
                        raise
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

    contact_map = np.zeros((m, m)).astype(dtype)
    t0 = time.time()
    for n in range(N):
        if verbose:
            prcnt_done = n/N * 100
            t = time.time() - t0
            if prcnt_done % 5 == 0:
                print(f'{prcnt_done}%')
        for i in range(m):
            for j in range(i+1):
                dist = np.linalg.norm(xyz[n, i, :] -xyz[n, j, :])
                if dist <= cutoff_distance:
                    contact_map[i,j] += 1
                    contact_map[j,i] += 1

    return contact_map


def xyz_to_distance(xyz, verbose = False):
    N, m, _ = xyz.shape
    D = np.zeros((N, m, m), dtype = np.float32)
    for i in range(N):
        if verbose:
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


def calculate_rg(xyz, verbose=False):
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
