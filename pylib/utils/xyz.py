import numpy as np
import time
import os.path as osp
import csv

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

