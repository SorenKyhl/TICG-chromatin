import csv
import os.path as osp

import h5py
import numpy as np
from sklearn.cluster import KMeans

from OpenMiChroM.ChromDynamics import MiChroM
from OpenMiChroM.Optimization import CustomMiChroMTraining


def get_k_means_seq(y, k):
    m, _ = y.shape
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(y)
    labels = [str(i) for i in kmeans.labels_]
    labels = list(map(lambda x: x.replace('1', 'A2'), labels))
    labels = list(map(lambda x: x.replace('0', 'A1'), labels))
    return labels

def run(y, dir):
    sim = MiChroM(temperature=1.0, time_step=0.01)
    sim.setup(platform="cuda")
    sim.saveFolder(osp.join(dir, 'michrom'))

def main():
    dir = '/home/erschultz/sequences_to_contact_maps/dataset_07_20_22/samples/sample7'
    y = np.load(osp.join(dir, 'y_diag.npy'))
    m = len(y)
    # labels = get_k_means_seq(y, 2)
    # print(labels)
    #
    # # write ChromSeq
    # bead_file = osp.join(dir, 'beads.txt')
    # with open(bead_file, 'w', newline = '') as f:
    #     wr = csv.writer(f, delimiter = '\t')
    #     wr.writerows(zip(np.arange(1, m+1), labels))


    # sim = CustomMiChroMTraining(bead_file)
    # filename = sys.path[0] + '/training/test_0.cndb'
    # mode = 'r'
    # myfile = h5py.File(filename, 'mode')
    # print("Calculating probabilities for 10 frames...")
    # for i in range(1,10):
    #     tl = np.array(myfile[str(i)])
    #     b.probCalculation_IC(state=tl)
    #     b.probCalculation_types(state=tl)
    # print('Getting parameters for Types and IC...')
    # ty = b.getLamb(exp_map=sys.path[0] + '/training/c18_10.dense')
    # ic = b.getLamb_types(exp_map=sys.path[0] + '/training/c18_10.dense')
    # print('Finished')

if __name__ == '__main__':
    main()
