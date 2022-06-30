import argparse
import csv
import os.path as osp
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--contact_map', type=str,
                        help='filepath to contact map')
    parser.add_argument('--verbose', action='store_true',
                        help='true for verbose mode')
    parser.add_argument('--mode', type=str,
                        help='{"plaid", "diag", "both"}')
    parser.add_argument('--constant', action='store_true',
                        help='True to include average contact frequency')
    parser.add_argument('--diag_bins', type=int,
                        help='number of diagonal bins')

    args = parser.parse_args()
    return args

def get_diag_goal(y, bins):
    m, _ = y.shape
    y_max = np.max(y)
    binsize = m / bins

    measure = []
    for b in range(bins):
        mask = np.zeros_like(y) # use mask to compute weighted average
        for i in range(m):
            for j in range(m):
                if int((i-j)/binsize) == b:
                    mask[i,j] = 1
                    mask[j,i] = 1
                    if i == j:
                        mask[i,j] = 2
        measure.append(np.sum((mask*y).flatten()))

    measure = np.array(measure / y_max)
    return measure

def get_plaid_goal(y, args):
    obj_goal = []
    y_max = np.max(y)
    if args.verbose:
        print('y_max: ', y_max)
        print(y, y.shape, y.dtype, '\n')

    x = np.load('x.npy')
    _, k = x.shape
    for i in range(k):
        seqi = x[:,i]
        if args.verbose:
            print(f'\ni={i}', seqi)
        for j in range(k):
            if j < i:
                # don't double count
                continue
            seqj = x[:,j]
            if args.verbose:
                print(f'\tj={j}', seqj)
            result = seqi @ y @ seqj
            obj_goal.append(result)

    obj_goal = np.array(obj_goal)
    obj_goal /= y_max # convert from freq to prob
    return obj_goal

def main():
    '''
    Calculate goal observables from contact map.

    Currently obs_goal_diag is not suported.
    '''
    args = getArgs()
    print(args)
    if args.mode == 'none':
        return

    if args.contact_map.endswith('.npy'):
        y = np.load(args.contact_map)
    elif args.contact_map.endswith('.txt'):
        y = np.loadtxt(args.contact_map)
    else:
        raise Exception(f"contact map format not recognized: {args.contact_map}")
    y = y.astype(float) # ensure float

    file = "seq0.txt"
    if osp.exists(file):
        seqi = np.loadtxt(file)
        m = len(seqi)
        y = y[:m, :m] # crop to m

    if args.mode == 'both':
        plaid_goal = get_plaid_goal(y, args)
        diag_goal = get_diag_goal(y, args.diag_bins)
    elif args.mode == 'plaid':
        plaid_goal = get_plaid_goal(y, args)
        diag_goal = np.zeros(20)
    elif args.mode == 'diag':
        plaid_goal = np.zeros(1) # shouldn't matter what goes here
        diag_goal = get_diag_goal(y, args.diag_bins)

    constant_goal = None
    if args.constant:
        constant_goal = np.sum(y)/np.max(y)
        with open('obj_goal_constant.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow(constant_goal)

    if args.verbose:
        print('plaid_goal: ', plaid_goal)
        print('diag_goal: ', diag_goal)
        print('constant_goal: ', constant_goal)

    with open('obj_goal.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(plaid_goal)

    with open('obj_goal_diag.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(diag_goal)

def test_plaid():
    sample_path='/home/erschultz/dataset_test/samples/sample1'
    offset = 0
    y = np.load(osp.join(sample_path, 'y.npy'))
    m = len(y)
    y_max = np.max(y)
    print('y_max: ', y_max)
    y = y.astype(float)

    print(y, y.shape, y.dtype, '\n')

    x = np.load(osp.join(sample_path, 'x.npy'))
    x = x[:m, :]
    x = x.astype(float)
    _, k = x.shape
    print(x, x.shape, x.dtype, '\n')

    # sorens method
    obj_goal = []
    for i in range(k):
        # seqi = x[:, i]
        seqi = np.loadtxt(osp.join(sample_path, f"seq{i}.txt"))[:m]
        for j in range(k):
            if j < i:
                # don't double count
                continue
            seqj = np.loadtxt(osp.join(sample_path, f"seq{i}.txt"))[:m]
            # result = seqi @ y @ seqj
            outer = np.outer(seqi, seqj)
            result = np.sum(outer * y)

            result /= y_max # convert to probability
            obj_goal.append(result)
    print(np.round(obj_goal, 7))

def test_diag():
    args = getArgs()
    dir = '/home/erschultz/dataset_test/samples/sample1'
    y = np.load(osp.join(dir, 'y.npy'))
    goal = get_diag_goal(y, 10)
    print(goal)

def test_constant():
    args = getArgs()
    dir = '/home/erschultz/dataset_test/samples/sample1'
    y = np.load(osp.join(dir, 'y.npy'))
    print(np.sum(y)/np.max(y))

if __name__ == '__main__':
    main()
    # test_constant()
    # test2()
