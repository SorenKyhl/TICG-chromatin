import csv
import os.path as osp
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_array


def xyz_load(xyz_filepath, delim = '\t', multiple_timesteps = True, save = False,
            N_min = None, N_max = None, down_sampling = 1):
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
                    xyz_timestep[i, :] = [float(j) for j in line[1:4]]
                    if i == N-1:
                        xyz.append(xyz_timestep)
                        xyz_timestep=np.empty((N, 3))

        xyz = np.array(xyz)
        if save:
            np.save(xyz_npy_file, xyz)
    if not multiple_timesteps:
        xyz = xyz[0]
    if N_min is None:
        N_min = 0
    if N_max is None:
        N_max = len(xyz)
    xyz = xyz[N_min:N_max:down_sampling]
    tf = time.time()
    print(f'Loaded xyz with shape {xyz.shape}')
    print(f'time: {np.round(tf - t0, 3)} s')
    return xyz

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
