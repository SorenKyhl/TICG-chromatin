import sys
import os.path as osp
import numpy as np

def main():
    x = np.load("../sequences_to_contact_maps/dataset_04_18_21/samples/sample40/x.npy")
    print(x)

    x0 = np.loadtxt("../sequences_to_contact_maps/dataset_04_18_21/samples/sample40/ground_truth2/k2/resources/seq0.txt")
    x1 = np.loadtxt("../sequences_to_contact_maps/dataset_04_18_21/samples/sample40/ground_truth2/k2/resources/seq1.txt")
    print(np.array_equal(x[:, 0] , x0))
    print(np.array_equal(x[:, 1] , x1))

if __name__ == '__main__':
    main()
