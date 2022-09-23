import argparse
import os
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--mode', type=str, help='{plaid, diag, both, all}')
    parser.add_argument('--k', type=int, help='deprecated')

    args = parser.parse_args()
    return args

def main():
    args = getArgs()
    print(args)
    assert args.mode in {'plaid', 'diag', 'both', 'all'}, 'invalid mode'

    # convergence plot
    convergence = np.loadtxt('convergence.txt')
    print(convergence)
    converged_it = None
    for i in range(1, len(convergence)):
        diff = convergence[i] - convergence[i-1]
        if np.abs(diff) < 1e-2 and convergence[i] < convergence[0]:
            converged_it = i
            break
    print('converged_it:', converged_it)

    plt.plot(convergence)
    if converged_it is not None:
        plt.axvline(converged_it, color = 'k', label = 'converged')
        plt.legend()
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.savefig("pconvergence.png")
    plt.close()

    if args.mode in {'plaid', 'both', 'all'}:
        # chis plot
        chis = np.loadtxt('chis.txt')
        if chis.ndim < 2:
            chis = np.atleast_2d(chis).T

        k = sympy.Symbol('k')
        result = sympy.solvers.solve(k*(k-1)/2 + k - chis.shape[1])
        k = np.max(result) # discard negative solution

        counter = 0
        for i in range(k):
            for j in range(k):
                if j < i:
                    continue
                chistr = "chi{}{}".format(LETTERS[i], LETTERS[j])
                plt.plot(chis[1:, counter], label = chistr)
                counter += 1
        plt.xlabel('Iteration')
        plt.ylabel('chi value')
        plt.legend(loc=(1.04,0))
        plt.tight_layout()
        plt.savefig("pchis.png")
        plt.close()

    if args.mode in {'diag', 'both', 'all'}:
        # diag chis plot
        diag_chis = np.loadtxt('chis_diag.txt')
        _, k = diag_chis.shape
        cmap = matplotlib.cm.get_cmap('tab20')
        ind = np.arange(k) % cmap.N
        colors = plt.cycler('color', cmap(ind))

        for i, c in enumerate(colors):
            plt.plot(diag_chis[:, i], label = i, color = c['color'])
        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Diagonal Parameter Value", fontsize=16)
        if k < 30:
            plt.legend(loc=(1.04,0))
        plt.tight_layout()
        plt.savefig("pchis_diag.png")
        plt.close()

    if args.mode == 'all':
        # constant chi plot
        constant_chi = np.loadtxt('chi_constant.txt')
        plt.plot(constant_chi)
        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("chi_constant value", fontsize=16)
        plt.tight_layout()
        plt.savefig("pchi_constant.png")
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
