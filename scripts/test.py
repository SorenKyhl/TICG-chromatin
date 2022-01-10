import sys
import os
import os.path as osp

import numpy as np

# ensure that I can find contact_map
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from makeLatexTable import METHODS
from get_seq import relabel_seq

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.dataset_classes import make_dataset

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def find_mising_ids():
    ids = set(range(1, 2001))
    dir = "/project2/depablo/erschultz/dataset_10_27_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            data_out_path = osp.join(dir, file, 'data_out')
            x_path = osp.join(dir, file, 'x.npy')
            if osp.exists(data_out_path) and osp.exists(x_path):
                ids.remove(id)

    print(ids, len(ids))

def upper_traingularize_chis():
    dir = "/project2/depablo/erschultz/dataset_08_26_21"
    samples = make_dataset(dir)
    for file in samples:
        file_dir = osp.join(dir, file)
        chis = np.load(osp.join(file_dir, 'chis.npy'))
        chis = np.triu(chis)

        np.savetxt(osp.join(file_dir, 'chis.txt'), chis, fmt='%0.5f')
        np.save(osp.join(file_dir, 'chis.npy'), chis)

def write_psi():
    dir = "/project2/depablo/erschultz/dataset_12_12_21/samples"
    # dir = "/home/eric/sequences_to_contact_maps/dataset_11_03_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
              #
            # x_linear = np.load(osp.join(file_dir, 'x_linear.npy'))
            # m, k = x_linear.shape
            # seq = np.zeros((m ,k))
            # for i in range(k):
            #     seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
            #     seq[:, i] = seq_i
            # if not np.array_equal(seq, x_linear):
            #     ids_to_check.add(int(file[6:]))
            #     print('fail2')
            #     passed = False
            file_dir = osp.join(dir, file)
            x_linear_file = osp.join(file_dir, 'x_linear.npy')
            xfile = osp.join(file_dir, 'x.npy')
            x = np.load(xfile)
            m, k = x.shape
            seq = np.zeros((m ,k))
            for i in range(k):
                seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                seq[:, i] = seq_i
            if not np.array_equal(seq, x):
                ids_to_check.add(int(file[6:]))
                print(int(file[6:]))

            if osp.exists(x_linear_file):
                x_linear = np.load(x_linear_file)
                np.save('psi.npy', x_linear)

def check_seq(dataset):
    # dir = "/project2/depablo/erschultz"
    dir = '/home/eric/sequences_to_contact_maps'
    if dataset is None:
        datasets = os.listdir(dir)
    else:
        datasets = [dataset]
    print(datasets)
    for dataset in datasets:
        if dataset.startswith("dataset") and osp.isdir(osp.join(dir, dataset)):
            ids_to_check = set()
            print(dataset)
            dataset_samples = osp.join(dir, dataset, 'samples')
            for file in os.listdir(dataset_samples):
                passed = True
                if file.startswith('sample'):
                    file_dir = osp.join(dataset_samples, file)
                    x = np.load(osp.join(file_dir, 'x.npy'))
                    m, k = x.shape
                    seq = np.zeros((m ,k))
                    for i in range(k):
                        seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                        seq[:, i] = seq_i
                    if not np.array_equal(seq, x):
                        ids_to_check.add(int(file[6:]))
                        print('fail1')
                        passed = False
                        # np.save(osp.join(file_dir, 'x.npy'), seq)

                    x_linear = np.load(osp.join(file_dir, 'x_linear.npy'))
                    m, k = x_linear.shape
                    seq = np.zeros((m ,k))
                    for i in range(k):
                        seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                        seq[:, i] = seq_i
                    if not np.array_equal(seq, x_linear):
                        ids_to_check.add(int(file[6:]))
                        print('fail2')
                        passed = False

                    if dataset.startswith("dataset_11_03"):
                    # if int(file[6:]) > 10:
                        row_sum = np.sum(x_linear[:, [0,1,3]], axis = 1)
                        if not np.all(row_sum <= 1):
                            ids_to_check.add(int(file[6:]))
                            passed = False
                            print('fail3')

                    print(f'{file} passed: {passed}')

            print(sorted(ids_to_check))

def makeDirsForMaxEnt(dataset, sample):
    sample_folder = osp.join('../sequences_to_contact_maps', dataset, 'samples', 'sample{}'.format(sample))
    assert osp.exists(sample_folder)

    for method in ['ground-truth', 'ground_truth-S', 'PCA', 'k_means', 'nmf', 'GNN-44-S']:
        os.mkdir(osp.join(sample_folder, method), mode = 0o755)
        for k in [2, 4]:
            os.mkdir(osp.join(sample_folder, method, f'k{k}'), mode = 0o755)
            for replicate in [1]:
                os.mkdir(osp.join(sample_folder, method, f'k{k}', f'replicate{replicate}'), mode = 0o755)

def main():
    dir = '/home/eric/sequences_to_contact_maps/dataset_08_29_21/samples/sample40'

    x = np.load(osp.join(dir, 'x.npy'))
    x_cluster = np.load(osp.join(dir, 'cluster', 'x.npy'))
    print(np.array_equal(x, x_cluster))

    for i in range(2):
        seq = np.loadtxt(osp.join(dir, f'seq{i}.txt'))
        seq_cluster = np.loadtxt(osp.join(dir, 'cluster', f'seq{i}.txt'))
        print(np.array_equal(seq, seq_cluster))

if __name__ == '__main__':
    write_psi()
    # find_mising_ids()
    # check_seq('dataset_11_03_21')
    # upper_traingularize_chis()
    # makeDirsForMaxEnt("dataset_08_29_21", 40)
