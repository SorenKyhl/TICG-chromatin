import os
import os.path as osp
import shutil
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
from pylib.Pysim import Pysim
from pylib.utils import default
from pylib.utils.xyz import xyz_load

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.plotting_utils import plot_xyz_gif

DIR = '/home/erschultz/sci_comm'
ROOT = osp.join(DIR, 'simulation')

def simulation():
    m = 100
    config = default.bonded_config.copy()
    config['bond_length'] = 140
    config['target_volume'] = 5
    config['grid_size'] = 140
    config['beadvol'] = 130000

    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['lmatrix_on'] = False
    config['dmatrix_on'] = False
    config['smatrix_on'] = False
    config['dump_frequency'] = 1000
    config['nbeads'] = m


    if osp.exists(ROOT):
        shutil.rmtree(ROOT)
    os.mkdir(ROOT, mode=0o755)


    stdout = sys.stdout
    with open(osp.join(ROOT, 'log.log'), 'w') as sys.stdout:
        sim = Pysim(ROOT, config, None, None, randomize_seed = True,
                    mkdir = False)
        t = sim.run_eq(10000, 300000, 1)
        print(f'Simulation took {np.round(t, 2)} seconds')

    sys.stdout = stdout

def make_hist_gif():
    file = osp.join(ROOT, 'production_out/output.xyz')
    m=100
    a = 10
    b = 30
    c = 60
    xyz = xyz_load(file, multiple_timesteps=True, N_max=None)[::, :m, :]
    print(xyz.shape)

    N = len(xyz)
    dists = np.zeros(N)
    dists2 = np.zeros(N)
    for i, xyz_i in enumerate(xyz):
        dist = np.linalg.norm(xyz_i[a] - xyz_i[b])
        dists[i] = dist
        # dist2 = np.linalg.norm(xyz_i[a] - xyz_i[c])
        # dists2[i] = dist2
    print(dists)

    file_dir = osp.join(DIR, 'hist_files')
    files = [osp.join(file_dir, f'hist_{i}.png') for i in range(N)]
    if not osp.exists(file_dir):
        os.mkdir(file_dir)

        for i, file in enumerate(files):
            fig = plt.figure()
            n_bins = max(10, min(50, i//2))
            if i < 20:
                y_max = 4
            elif i < 120:
                y_max = 8
            elif i < 160:
                y_max = 10
            elif i < 200:
                y_max = 12
            elif i < 250:
                y_max = 14
            else:
                y_max = 16
            n, bins, patches = plt.hist(dists[:i],
                                        bins = n_bins,
                                        alpha = 0.5, label = 'Close')
            # n, bins, patches = plt.hist(dists2[:i], weights = weights,
            #                             bins = n_bins,
            #                             alpha = 0.5, label = 'Far')
            plt.ylabel('Number of Structures', fontsize=16)
            plt.xlabel('Distance', fontsize=16)
            plt.xlim(0, 1300)
            plt.ylim(0, y_max)
            # plt.legend(loc='upper right')
            plt.savefig(file)
            plt.close()

    # build gif
    frames = []
    for f in files:
        frames.append(imageio.imread(f))

    imageio.mimsave(osp.join(DIR, 'hist.gif'), frames, format='GIF', fps=5)


def make_xyz_gif():
    file = osp.join(ROOT, 'production_out/output.xyz')
    m=100
    k=2

    i = 10
    j = 30

    x = np.zeros((m,k))
    x[i, 0] = 1
    x[j, 1] = 1
    xyz = xyz_load(file, multiple_timesteps=True)[::, :m, :]
    print(xyz.shape)

    plot_xyz_gif(xyz, x, DIR, colors = ['red', 'green'], fps = 5)


def main():
    # simulation()
    make_xyz_gif()
    # make_hist_gif()

if __name__ == '__main__':
    main()
