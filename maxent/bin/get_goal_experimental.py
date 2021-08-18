import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import csv

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--m', type=int, default=1024, help='number of particles, will crop seq{i}.txt and y')
    parser.add_argument('--k', type=int, help='number of particle types (inferred from chi if None)')
    parser.add_argument('--contact_map', type=str, help='filepath to contact map')
    parser.add_argument('--verbose', action='store_true', help='true for verbose mode')

    args = parser.parse_args()
    return args

def main():
    '''
    Calculate goal observables from contact map.

    Currently obs_goal_diag is not suported.
    '''
    args = getArgs()
    obj_goal = []

    if args.contact_map.endswith('.npy'):
        y = np.load(args.contact_map)
    elif args.contact_map.endswith('.txt'):
        y = np.loadtxt(args.contact_map)
    else:
        raise Exception("contact map format not recognized: {}".format(args.contact_map))

    y = y[:args.m, :args.m] # crop to m
    y = np.triu(y) # avoid double counting due to symmetry
    y_max = np.max(y)
    y_ones = np.ones_like(y)
    y_ones = np.triu(y_ones)
    if args.verbose:
        print(y)

    for i in range(args.k):
        seqi = np.loadtxt("seq{}.txt".format(i))[:args.m]
        if args.verbose:
            print('\ni={}'.format(i), seqi)
        for j in range(args.k):
            if j < i:
                # don't double count
                continue
            seqj = np.loadtxt("seq{}.txt".format(j))[:args.m]
            if args.verbose:
                print('\tj={}'.format(j), seqj)
            result = seqi @ y @ seqj
            denom = seqi @ y_ones @ seqj
            result /= denom # take average
            result /= y_max # convert from freq to prob
            obj_goal.append(result)

    with open('obj_goal.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(obj_goal)
    with open('obj_goal_diag.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(np.zeros(20))

if __name__ == '__main__':
    main()
