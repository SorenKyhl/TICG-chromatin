import argparse
import os
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--mode', type=str, help='{plaid, diag, both}')
    parser.add_argument('--k', type=int, help='number of particle types')

    args = parser.parse_args()
    return args

def main():
    args = getArgs()
    assert args.mode in {'plaid', 'diag', 'both'}

    # convergence plot
    convergence = np.loadtxt('convergence.txt')
    plt.plot(convergence)
    plt.xlabel('Iteration')
    plt.savefig("pconvergence.png")
    plt.close()

    if osp.exists('convergence_diag.txt'):
        convergence = np.loadtxt('convergence_diag.txt')
        plt.plot(convergence)
        plt.xlabel('Iteration')
        plt.savefig("pconvergence_diag.png")
        plt.close()

    if args.mode in {'plaid', 'both'}:
        # chis plot
        chis = np.loadtxt('chis.txt')
        if chis.ndim < 2:
            chis = np.atleast_2d(chis).T
        counter = 0
        for i in range(args.k):
            for j in range(args.k):
                if j < i:
                    continue
                chistr = "chi{}{}".format(LETTERS[i], LETTERS[j])
                plt.plot(chis[1:, counter], label = chistr)
                counter += 1
        plt.xlabel('Iteration')
        plt.ylabel('chi value')
        plt.legend()
        plt.savefig("pchis.png")
        plt.close()

    if args.mode in {'diag', 'both'}:
        # diag chis plot
        diag_chis = np.loadtxt('chis_diag.txt')
        _, k = diag_chis.shape
        cmap = matplotlib.cm.get_cmap('tab20')
        ind = np.arange(k) % cmap.N
        colors = plt.cycler('color', cmap(ind))

        for i, c in enumerate(colors):
            plt.plot(diag_chis[:, i], label = i, color = c['color'])
        plt.xlabel("Iteration")
        plt.ylabel("chi_diagonal value")
        plt.legend(loc=(1.04,0))
        plt.tight_layout()
        plt.savefig("pchis_diag.png")
        plt.close()


def test():
    args = getArgs()
    args.k = 1
    dataset = '/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/'
    diag_chis = np.loadtxt(osp.join(dataset, 'sample1/PCA-normalize-diagOn/k2/replicate1/chis_diag.txt'))
    _, k = diag_chis.shape

    cmap = matplotlib.cm.get_cmap('tab20')
    ind = np.arange(k) % cmap.N
    colors = plt.cycler('color', cmap(ind))

    for i, c in enumerate(colors):
        plt.plot(diag_chis[:, i], label = i, color = c['color'])
    plt.xlabel("Iteration")
    plt.ylabel("chi_diagonal value")
    plt.legend(loc=(1.04,0))
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    # test()
