import json
import numpy as np
import sys
import argparse

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

    allchis = np.loadtxt('chis.txt')
    letters='ABCDEFG'

    # get last row of 'chis.txt'
    lastchis = list(allchis[int(args.it)])

    counter = 0
    for i in range(args.k):
        for j in range(args.k):
            if j < i:
                continue
            key = 'chi{}{}'.format(letters[i], letters[j])
            config[key] = lastchis[counter]
            counter += 1

    with open(config_file, "w") as f:
        json.dump(config, f)

def test():
    args = getArgs()
    args.k = 2
    args.it = 0

    config = {}

    allchis = np.loadtxt('maxent/resources/chis.txt')
    print(allchis)
    letters='ABCDEFG'

    # get last row of 'chis.txt'
    lastchis = list(allchis[int(args.it)])

    counter = 0
    for i in range(args.k):
        for j in range(args.k):
            if j < i:
                continue
            key = 'chi{}{}'.format(letters[i], letters[j])
            config[key] = lastchis[counter]
            counter += 1

    print(config)

if __name__ == '__main__':
    main()
