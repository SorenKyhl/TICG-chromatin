import json
import numpy as np
import sys
import argparse

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--it', type=int, help='current iteration')
    parser.add_argument('--k', type=int, help='number of particle types')

    args = parser.parse_args()
    return args

def main():
    args = getArgs()

    config_file = "iteration{}/config.json".format(args.it)

    with open(config_file, "r") as f:
        config = json.load(f)

    allchis = np.atleast_2d(np.loadtxt('chis.txt'))
    #if len(allchis[0] == 0):
        # shape will be wrong if k = 1
        #allchis = allchis.T

    # get last row of 'chis.txt'
    try:
        lastchis = list(allchis[int(args.it)])
        #lastchis = list(allchis[-1])
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
            print( i, j, counter)

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
