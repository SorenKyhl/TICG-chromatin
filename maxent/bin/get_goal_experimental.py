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
    parser.add_argument('--diag_bins', type=int,
                        help='number of diagonal bins')
    parser.add_argument('--v_bead', type=int, default=520,
                        help='volume of bead')
    parser.add_argument('--grid_size', type=float, default=28.7,
                        help='side length of grid cell')

    args = parser.parse_args()
    return args

def get_diag_goal(y, args):
    '''Input y should have max 1.'''
    m, _ = y.shape
    binsize = m / args.diag_bins

    obj_goal_diag = []
    for b in range(args.diag_bins):
        mask = np.zeros_like(y) # use mask to compute weighted average
        for i in range(m):
            for j in range(m):
                if int((i-j)/binsize) == b:
                    mask[i,j] = 1
                    mask[j,i] = 1
                    if i == j:
                        mask[i,j] = 2
        obj_goal_diag.append(np.sum((mask*y).flatten()))

    obj_goal_diag = np.array(obj_goal_diag)
    obj_goal_diag *= (args.v_bead / args.grid_size**3)
    return obj_goal_diag

def get_plaid_goal(y, args, x = None):
    '''Input y should have max 1.'''
    obj_goal = []
    if args.verbose:
        print(y, y.shape, y.dtype, '\n')

    if x is None:
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
    obj_goal *= (args.v_bead / args.grid_size**3)

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
    y /= np.max(y)

    file = "seq0.txt"
    if osp.exists(file):
        seqi = np.loadtxt(file)
        m = len(seqi)
        y = y[:m, :m] # crop to m

    constant_goal = None
    plaid_goal = None
    diag_goal = None
    if args.mode == 'both':
        plaid_goal = get_plaid_goal(y, args)
        diag_goal = get_diag_goal(y, args)
    elif args.mode == 'plaid':
        plaid_goal = get_plaid_goal(y, args)
    elif args.mode == 'diag':
        diag_goal = get_diag_goal(y, args)
    elif args.mode == 'all':
        plaid_goal = get_plaid_goal(y, args)
        diag_goal = get_diag_goal(y, args)
        constant_goal = np.sum(y)/np.max(y)
        constant_goal *= (args.v_bead / args.grid_size**3)
        with open('obj_goal_constant.txt', 'w') as f:
            f.write(f'{constant_goal}')

    if args.verbose:
        print('plaid_goal: ', plaid_goal)
        print('diag_goal: ', diag_goal)
        print('constant_goal: ', constant_goal)

    if plaid_goal is not None:
        with open('obj_goal.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow(plaid_goal)

    if diag_goal is not None:
        with open('obj_goal_diag.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow(diag_goal)

def test_plaid():
    args = getArgs()
    dir = '/home/erschultz/dataset_test/samples/sample1'
    y = np.load(osp.join(dir, 'y.npy'))
    y = y.astype(float)
    y /= np.max(y)
    # print(y.shape, y.dtype, '\n')

    x = np.load(osp.join(dir, 'x.npy'))
    x = x.astype(float)
    m, k = x.shape
    # print(x.shape, x.dtype, '\n')

    # obj_goal = get_plaid_goal(y, args, x)
    # print(obj_goal)

    x = np.ones(m).reshape(-1, 1)
    obj_goal = get_plaid_goal(y, args, x)
    print(obj_goal)


def test_diag():
    args = getArgs()
    args.diag_bins = 10
    dir = '/home/erschultz/dataset_test/samples/sample1'
    y = np.load(osp.join(dir, 'y.npy'))
    y = y.astype(float)
    y /= np.max(y)

    # goal = get_diag_goal(y, args)
    # print(goal)
    # goal = get_goal_diag_soren(y, 10)
    # print(goal)

def test_constant():
    args = getArgs()
    dir = '/home/erschultz/dataset_test/samples/sample1'
    y = np.load(osp.join(dir, 'y.npy'))
    y = y.astype(float)
    y /= np.max(y)
    print(np.sum(y))

if __name__ == '__main__':
    main()
    # test_constant()
    # test_plaid()
    # test_diag()
