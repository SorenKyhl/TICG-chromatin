import sys

from pylib.utils.plotting_utils import plot_matrix

sys.path.append('/home/erschultz/TICG-chromatin')
from max_ent import fit, setup_config
from scripts.import_contactmap_straw import *


def main():
    fname = "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic"
    odir = '/home/erschultz/timing_analysis'
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    kb=25600
    resolutions = [5, 1]
    m_list = []
    for i in range(len(resolutions)-1):
        print(i)
        resolution = resolutions[i]
        m = kb / resolution
        print(m)
        m = int(m)
        m_list.append(m)
        single_experiment_dataset(fname, osp.join('timing_analysis', str(m)),
                                resolution * 1000, m, chroms=[1,2], seed=23)

    for m in m_list:
        print(m)
        samples_dir = osp.join(odir, str(m), 'samples')
        for i in range(1, 16):
            s_dir = osp.join(odir, str(m), 'samples', f'sample{i}')
            y = np.load(osp.join(s_dir, 'y.npy'))
            plot_matrix(y, osp.join(s_dir, 'y.png'), vmax='mean')


if __name__ == '__main__':
    main()
