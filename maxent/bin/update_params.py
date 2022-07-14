import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import sympy

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--it', type=int,
                        help='current iteration')
    parser.add_argument('--mode', type=str,
                        help='update mode')

    args = parser.parse_args()
    return args

def main():
    args = getArgs()

    config_file = "iteration{}/config.json".format(args.it)

    with open(config_file) as f:
        config = json.load(f)

    if args.mode in {'diag', 'both', 'all'}:
        allchis = np.loadtxt('chis_diag.txt')

        # get last row of 'chis_diag.txt'
        lastchis = list(allchis[int(args.it)])

        config['diag_chis'] = lastchis


    if args.mode in {'plaid', 'both', 'all'}:
        # read in current chis
        with open('chis.txt', "r") as f_chis:
            lines = f_chis.readlines()
            current_chis = lines[args.it].split()
            current_chis = [float(x) for x in current_chis]
        # print("current chi values: ", current_chis)

        k = sympy.Symbol('k')
        result = sympy.solvers.solve(k*(k-1)/2 + k - len(current_chis))
        k = np.max(result) # discard negative solution

        counter = 0
        for i in range(k):
            for j in range(k):
                if j < i:
                    continue
                key = 'chi{}{}'.format(LETTERS[i], LETTERS[j])
                try:
                    config[key] = current_chis[counter]
                except IndexError as e:
                    print('Index Error')
                    print('lines:\n', lines)
                    print('current_chis:\n', current_chis, len(current_chis))
                    print('k: ', result, k)
                    raise
                counter += 1

    if args.mode == 'all':
        allchis = np.loadtxt('chi_constant.txt')
        print(allchis.shape)


    with open(config_file, "w") as f:
        json.dump(config, f)

def test():
    args = getArgs()
    args.k = 1
    args.it = 5

    config = {}

    allchis = np.atleast_2d(np.loadtxt('../sequences_to_contact_maps/dataset_08_26_21/samples/sample40/GNN-23/k1/chis.txt'))
    print(allchis)
    if len(allchis[0] == 0):
        allchis = allchis.T
        print(allchis.shape)

    # get last row of 'chis.txt'
    try:
        lastchis = list(allchis[int(args.it)])
        print(lastchis)
    except IndexError as e:
        print('Index Error')
        print('allchis:\n', allchis)
        raise

    counter = 0
    for i in range(args.k):
        for j in range(args.k):
            if j < i:
                continue
            key = 'chi{}{}'.format(LETTERS[i], LETTERS[j])
            config[key] = lastchis[counter]
            counter += 1

    print(config)

if __name__ == '__main__':
    main()
