import sys
import os
import os.path as osp

import numpy as np

# ensure that I can find contact_map
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from makeLatexTable import METHODS

def find_mising_ids():
    ids = set(range(1, 2001))
    dir = "/project2/depablo/erschultz/dataset_10_25_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            if osp.exists(osp.join(dir, file, 'data_out')):
                ids.remove(id)

    print(ids)

def upper_traingularize_chis():
    dir = "/project2/depablo/erschultz/dataset_10_25_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            file_dir = osp.join(dir, file)
            chis = np.load(osp.join(file_dir, 'chis.npy'))
            chis = np.triu(chis)

            np.savetxt(osp.join(file_dir, 'chis.txt'), chis, fmt='%0.5f')
            np.save(osp.join(file_dir, 'chis.npy'), chis)

def check_seq():
    ids_to_check = set()
    dir = "/project2/depablo/erschultz/dataset_10_25_21/samples"
    k = 4
    m = 1024
    for file in os.listdir(dir):
        if file.startswith('sample'):
            file_dir = osp.join(dir, file)
            x = np.load(osp.join(file_dir, 'x.npy'))
            seq = np.zeros((m ,k))
            for i in range(k):
                seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                seq[:, i] = seq_i
            if not np.array_equal(seq, x):
                ids_to_check.add(int(file[6:]))
                np.save(osp.join(file_dir, 'x.npy'), seq)

    print(sorted(ids_to_check))

def makeDirsForMaxEnt(dataset, sample):
    sample_folder = osp.join('../sequences_to_contact_maps', dataset, 'samples', 'sample{}'.format(sample))
    assert osp.exists(sample_folder)

    for method in METHODS:
        os.mkdir(osp.join(sample_folder, method), mode = 0o755)
        for k in [2, 4, 6]:
            os.mkdir(osp.join(sample_folder, method, 'k{}'.format(k)), mode = 0o755)

if __name__ == '__main__':
    # find_mising_ids()
    # check_seq()
    upper_traingularize_chis()
    # makeDirsForMaxEnt("dataset_08_29_21", 40)
