import argparse
import csv
import json
import math
import os.path as osp
import sys
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--contact_map', type=str,
                        help='filepath to contact map')
    parser.add_argument('--verbose', action='store_true',
                        help='true for verbose mode')
    parser.add_argument('--mode', type=str,
                        help='{"plaid", "diag", "both"}')
    parser.add_argument('--v_bead', type=int, default=520,
                        help='volume of bead')
    parser.add_argument('--grid_size', type=float, default=28.7,
                        help='side length of grid cell')

    args = parser.parse_args()
    return args

def get_diag_goal(y, args, return_correction = False):
    '''Input y should have max 1.'''
    m, _ = y.shape
    if args.diag_cutoff > m:
        args.diag_cutoff = m

    obj_goal_diag = []
    correction = []
    for bin in range(args.diag_bins):
        mask = get_mask(bin, m, args.diag_bins, args.dense, args.n_small_bins,
                        args.small_binsize, args.big_binsize, args.diag_start,
                        args.diag_cutoff)
        obj_goal_diag.append(np.sum((mask*y).flatten()))
        correction.append(np.sum(mask))

    obj_goal_diag = np.array(obj_goal_diag)
    obj_goal_diag *= (args.v_bead / args.grid_size**3)

    if return_correction:
        return obj_goal_diag, correction
    else:
        return obj_goal_diag

# @njit
def get_mask(bin, m, diag_bins, dense, n_small_bins,
            small_binsize, big_binsize, diag_start, diag_cutoff):
    mask = np.zeros((m, m)) # use mask to compute weighted average
    if dense:
        dividing_line = n_small_bins * small_binsize
    else:
        binsize = m / diag_bins

    for i in range(m):
        for j in range(i+1):
            d = i-j # don't need abs
            if d < diag_start or d > diag_cutoff:
                continue
            d_eff = d - diag_start
            if dense:
                if d_eff > dividing_line:
                    curr_bin = n_small_bins + math.floor( (d_eff - dividing_line) / big_binsize)
                else:
                    curr_bin =  math.floor( d_eff / small_binsize)
            else:
                curr_bin = int(d_eff/binsize)
            if curr_bin == bin:
                mask[i,j] = 1
                mask[j,i] = 1
                if i == j:
                    mask[i,j] = 2

    return mask

def get_plaid_goal(y, args, x = None):
    '''Input y should have max 1.'''
    obj_goal = []
    if args.verbose:
        print(y, y.shape, y.dtype, '\n')
        if x is not None:
            print(x, x.shape, x.dtype, '\n')

    if x is None:
        x = np.load('x.npy')
    _, k = x.shape
    for i in range(k):
        seqi = x[:, i]
        # if args.verbose:
        #     print(f'\ni={i}', seqi)
        for j in range(k):
            if j < i:
                # don't double count
                continue
            seqj = x[:,j]
            # if args.verbose:
            #     print(f'\tj={j}', seqj)
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
    t0 = time.time()
    args = getArgs()
    print(args)
    if args.mode == 'none':
        return

    with open('config.json', 'r') as f:
        config = json.load(f)
        if config['dense_diagonal_on']:
            args.dense = True
            args.n_small_bins = config['n_small_bins']
            args.small_binsize = config['small_binsize']
            args.big_binsize = config['big_binsize']
        else:
            args.dense = False
            args.n_small_bins = None
            args.small_binsize = None
            args.big_binsize = None
        args.diag_bins = len(config['diag_chis'])
        args.diag_start = config['diag_start']
        args.diag_cutoff = config['diag_cutoff']
        args.m = config['nbeads']

    if not osp.exists(args.contact_map):
        dir = osp.split(args.contact_map)[0]

        obj_goal = osp.join(dir, 'obj_goal.txt')
        assert osp.exists(obj_goal), f"neither {args.contact_map} nor {osp.join(dir, 'obj_goal.txt')} exists"
        copyfile(obj_goal, 'obj_goal.txt')

        obj_goal_diag = osp.join(dir, 'obj_goal_diag.txt')
        assert osp.exists(obj_goal_diag)
        copyfile(obj_goal_diag, 'obj_goal_diag.txt')

        return

    if args.contact_map.endswith('.npy'):
        y = np.load(args.contact_map)
    elif args.contact_map.endswith('.txt'):
        y = np.loadtxt(args.contact_map)
    else:
        raise Exception(f"contact map format not recognized: {args.contact_map}")
    y = y.astype(float) # ensure float
    y /= np.mean(np.diagonal(y))

    y = y[:args.m, :args.m] # crop to m
    print('shape: ', y.shape)

    constant_goal = None
    plaid_goal = None
    diag_goal = None
    if args.mode == 'both':
        plaid_goal = get_plaid_goal(y, args)
        diag_goal = get_diag_goal(y, args)
        if args.verbose:
            print('plaid_goal: ', plaid_goal, plaid_goal.shape)
            print('diag_goal: ', diag_goal, diag_goal.shape)
    elif args.mode == 'plaid':
        plaid_goal = get_plaid_goal(y, args)
        if args.verbose:
            print('plaid_goal: ', plaid_goal, plaid_goal.shape)
    elif args.mode == 'diag':
        diag_goal = get_diag_goal(y, args)
        if args.verbose:
            print('diag_goal: ', diag_goal, diag_goal.shape)
    elif args.mode == 'all':
        plaid_goal = get_plaid_goal(y, args)
        diag_goal = get_diag_goal(y, args)
        constant_goal = np.sum(y)/np.max(y)
        constant_goal *= (args.v_bead / args.grid_size**3)
        with open('obj_goal_constant.txt', 'w') as f:
            f.write(f'{constant_goal}')
        if args.verbose:
            print('plaid_goal: ', plaid_goal, plaid_goal.shape)
            print('diag_goal: ', diag_goal, diag_goal.shape)
            print('constant_goal: ', constant_goal, constant_goal.shape)

    if plaid_goal is not None:
        with open('obj_goal.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow(plaid_goal)

    if diag_goal is not None:
        with open('obj_goal_diag.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow(diag_goal)

    tf = time.time()
    print('time: ', tf - t0)


class Tester():
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
        args.diag_bins = 32
        args.n_small_bins = 16
        args.small_binsize = 8
        args.n_big_bins = 16
        args.big_binsize = 24
        args.dense = True
        dir = '/home/erschultz/sequences_to_contact_maps/dataset_07_20_22/samples/sample4'
        y = np.load(osp.join(dir, 'y.npy'))
        y = y.astype(float)
        y /= np.max(y)
        m = len(y)
        print(m)

        t0 = time.time()
        goal = get_diag_goal(y, args)
        tf = time.time()
        # print(goal)
        print('time: ', tf - t0)
        # t0 = time.time()
        # goal_soren = soren_get_goal_diag(y, args.diag_bins,
                                # dense_diagonal_on = args.dense_diagonal_on)
        # tf = time.time()
        # print(goal_soren)
        # print('time: ', tf - t0)
        # assert np.allclose(goal, goal_soren)

    def test_constant():
        args = getArgs()
        dir = '/home/erschultz/dataset_test/samples/sample1'
        y = np.load(osp.join(dir, 'y.npy'))
        y = y.astype(float)
        y /= np.max(y)
        print(np.sum(y))

if __name__ == '__main__':
    main()
    # Tester.test_constant()
    # Tester.test_plaid()
    # Tester.test_diag()
