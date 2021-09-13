import sys
import os
import os.path as osp

import numpy as np

def find_mising_ids():
    ids = set(range(1, 2001))
    dir = "/project2/depablo/erschultz/dataset_08_29_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            ids.remove(id)

    print(ids)

def main():
    dir = '../sequences_to_contact_maps/dataset_08_26_21/samples'
    x = np.load(osp.join(dir, 'sample1201', 'x.npy'))
    print(x)

def check_seq():
    ids_to_check = set()
    dir = "/project2/depablo/erschultz/dataset_08_26_21/samples"
    # dir = '../sequences_to_contact_maps/dataset_08_26_21/samples'
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

if __name__ == '__main__':
    check_seq()
