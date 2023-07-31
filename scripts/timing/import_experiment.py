import sys

from pylib.utils.plotting_utils import plot_matrix

sys.path.append('/home/erschultz/TICG-chromatin')
from scripts.import_contactmap_straw import *


def main():
    fname = "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic"
    odir = '/home/erschultz/timing_analysis'
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    kb=25600
    resolutions = [50, 25, 10, 5, 1]
    for i in range(4):
        print(i)
        resolution = resolutions[i]
        m = kb / resolution
        print(m)
        m = int(m)
        single_experiment_dataset(fname, osp.join('timing_analysis', str(m)),
                                resolution * 1000, m, chroms=[1,2], seed=23)

def plot():
    dir = '/home/erschultz/timing_analysis'
    for m in [512, 1024, 2560, 5120]:
        print(m)
        samples_dir = osp.join(dir, str(m), 'samples')
        for i in range(1, 16):
            s_dir = osp.join(dir, str(m), 'samples', f'sample{i}')
            y = np.load(osp.join(s_dir, 'y.npy'))
            plot_matrix(y, osp.join(s_dir, 'y.png'), vmax='mean')




if __name__ == '__main__':
    # main()
    plot()
