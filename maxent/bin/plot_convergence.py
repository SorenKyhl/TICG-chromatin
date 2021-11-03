import os
import os.path as osp

import matplotlib.pyplot as plt
import argparse
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

    if args.mode == 'plaid' or args.mode == 'both':
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


    if args.mode == 'diag' or args.mode == 'both':
        raise Exception("diag chis not fully supported yet")
        convergence_diag = np.loadtxt('convergence_diag.txt')
        plt.plot(convergence_diag)
        plt.xlabel('Iteration')
        plt.savefig("pconvergence_diag.png")
        plt.close()

def test():
    args = getArgs()
    args.k = 1
    convergence = np.loadtxt('../sequences_to_contact_maps/dataset_08_26_21/samples/sample40/GNN-23/k1/convergence.txt')
    plt.plot(convergence)
    plt.xlabel('Iteration')
    plt.show()

    chis = np.loadtxt('../sequences_to_contact_maps/dataset_08_26_21/samples/sample40/GNN-23/k1/chis.txt')
    if chis.ndim < 2:
        # shape will be wrong if k = 1
        chis = np.atleast_2d(chis).T
        print(chis.shape)
    counter = 0
    for i in range(args.k):
        for j in range(args.k):
            if j < i:
                continue
            chistr = "chi{}{}".format(LETTERS[i], LETTERS[j])
            print(chis[:, counter])
            plt.plot(chis[:, counter], label = chistr)
            counter += 1
    plt.xlabel('Iteration')
    plt.ylabel('chi value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
