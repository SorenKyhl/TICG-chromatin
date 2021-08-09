import os
import os.path as osp
import sys

import numpy as np
import argparse

from sklearn.decomposition import PCA

sys.path.insert(1, '../sequences_to_contact_maps')
from neural_net_utils.utils import calculateDistanceStratifiedCorrelation

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    # '../../sequences_to_contact_maps/dataset_04_18_21'
    # "/project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--data_folder', type=str, default='/project2/depablo/erschultz/dataset_04_18_21', help='Location of input data')
    parser.add_argument('--sample', type=int, default=2, help='sample id')
    parser.add_argument('--method', type=str, default='random', help='method for assigning particle types')


    args = parser.parse_args()
    args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))
    args.output_folder = osp.join(args.sample_folder, args.method)
    if not osp.exists(args.output_folder):
        os.mkdir(args.output_folder, mode = 0o755)
    return args

# plotting functions
def plotDistanceStratifiedPearsonCorrelation(y, yhat, args):
    overall_corr, corr_arr = calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson')
    title = 'Overall Pearson R: {}'.format(np.round(overall_corr, 3))

    plt.plot(np.arange(args.m-1), corr_arr, color = 'black')
    plt.ylim(-0.5, 1)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
    plt.title(title, fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(args.output_folder, 'distance_pearson.png'))
    plt.close()

def main():
    args = getArgs()


if __name__ == '__main__':
    main()
