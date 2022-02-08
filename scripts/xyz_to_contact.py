import os.path as osp
import sys

import numpy as np
import json
import math



paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.xyz_utils import *
from plotting_functions import plotContactMap

def xyz_to_contact(xyz, grid_size):
    m, _ = xyz.shape
    contact_map = np.zeros((m,m))

    for i in range(m):
        grid_i = [d // grid_size for d in xyz[i]]
        for j in range(i+1):
            grid_j = [d // grid_size for d in xyz[j]]
            if grid_i == grid_j:
                contact_map[i,j] += 1
                contact_map[j,i] += 1

    return contact_map


def main():
    dir='/home/eric/dataset_test/samples/sample82'
    file = osp.join(dir, 'data_out/output.xyz')

    config_file = osp.join(dir, 'config.json')
    with open(config_file, 'rb') as f:
        config = json.load(f)
        grid_size = int(config['grid_size'])


    x = np.load(osp.join(dir, 'x.npy'))
    y = np.load(osp.join(dir, 'y.npy'))
    xyz = xyzLoad(file, multiple_timesteps=True)
    N, m, _ = xyz.shape
    print(N)
    overall = np.zeros((m,m))
    for i in range(N):
        contact_map = xyz_to_contact(xyz[i], grid_size)
        plotContactMap(contact_map, osp.join(dir, 'sc_contact', f'{i}.png'))
        overall += contact_map
    plotContactMap(overall, osp.join(dir, 'sc_contact', 'overall.png'))
    dif = overall - y
    plotContactMap(dif, osp.join(dir, 'sc_contact', 'dif.png'), cmap = 'blue-red')

    print(np.array_equal(y, overall))



if __name__ == '__main__':
    main()
