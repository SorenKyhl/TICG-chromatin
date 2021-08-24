import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

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

        # chis plot
        chis = np.loadtxt('chis.txt')
        letters = 'ABCDEFGHI'
        counter = 0
        for i in range(args.k):
            for j in range(args.k):
                if j < i:
                    continue
                chistr = "chi{}{}".format(letters[i], letters[j])
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
    args.k = 2

    convergence = np.loadtxt('../sequences_to_contact_maps/maxent_08_12_21/convergence.txt')
    plt.plot(convergence)
    plt.xlabel('Iteration')
    plt.show()
    plt.savefig("pconvergence.png")

    chis = np.loadtxt('../sequences_to_contact_maps/maxent_08_12_21/chis.txt')
    letters = 'ABCDEFGHI'
    counter = 0
    for i in range(args.k):
        for j in range(args.k):
            if j < i:
                continue
            chistr = "chi{}{}".format(letters[i], letters[j])
            plt.plot(chis[:, counter], label = chistr)
            counter += 1
    plt.xlabel('Iteration')
    plt.ylabel('chi value')
    plt.legend()
    plt.show()
    plt.savefig("pchis.png")


if __name__ == '__main__':
    main()
