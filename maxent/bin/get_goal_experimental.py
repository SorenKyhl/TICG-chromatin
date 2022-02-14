import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import csv
import time

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.argparseSetup import str2int


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--k', type=int, help='number of particle types')
    parser.add_argument('--contact_map', type=str, help='filepath to contact map')
    parser.add_argument('--verbose', action='store_true', help='true for verbose mode')
    parser.add_argument('--mode', type=str, help='{"plaid", "diag", "both"}')

    args = parser.parse_args()
    return args

def get_diag_goal(y, args):
    return np.zeros(20)

def get_plaid_goal(y, m, args):
    obj_goal = []
    y_max = np.max(y)
    if args.verbose:
        print('y_max: ', y_max)
        print(y, y.shape, y.dtype, '\n')
    for i in range(args.k):
        seqi = np.loadtxt(f"seq{i}.txt")
        if args.verbose:
            print(f'\ni={i}', seqi)
        for j in range(args.k):
            if j < i:
                # don't double count
                continue
            seqj = np.loadtxt(f"seq{j}.txt")
            if args.verbose:
                print(f'\tj={j}', seqj)
            result = seqi @ y @ seqj
            result /= m**2 # take average
            result /= y_max # convert from freq to prob
            obj_goal.append(result)

    return obj_goal

def main():
    '''
    Calculate goal observables from contact map.

    Currently obs_goal_diag is not suported.
    '''
    args = getArgs()

    if args.contact_map.endswith('.npy'):
        y = np.load(args.contact_map)
    elif args.contact_map.endswith('.txt'):
        y = np.loadtxt(args.contact_map)
    else:
        raise Exception(f"contact map format not recognized: {args.contact_map}")

    seqi = np.loadtxt("seq0.txt")
    m = len(seqi)
    y = y.astype(float) # ensure float
    y = y[:m, :m] # crop to m

    if args.mode == 'both':
        plaid_goal = get_plaid_goal(y, m, args)
        diag_goal = get_diag_goal(y, args)
    elif args.mode == 'plaid':
        plaid_goal = get_plaid_goal(y, m, args)
        diag_goal = np.zeros(20)
    elif args.mode == 'diag':
        plaid_goal = np.zeros(1) # shouldn't matter what goes here
        diag_goal = get_diag_goal(y, m, args)


    if args.verbose:
        print('obj_goal: ', plaid_goal)

    with open('obj_goal.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(plaid_goal)

    with open('obj_goal_diag.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(diag_goal)

def test():
    sample_path='../sequences_to_contact_maps/dataset_08_24_21/samples/sample1201'

    m = 1024
    offset = 0
    y = np.load(osp.join(sample_path, 'y.npy'))[:m, :m]
    # y = np.loadtxt(osp.join(sample_path, 'data_out', 'contacts.txt'))[:m, :m]
    y_max = np.max(y)
    print('y_max: ', y_max)
    # y = np.triu(y, k = offset) # avoid double counting due to symmetry
    y = y.astype(float)

    print(y, y.shape, y.dtype, '\n')

    x = np.load(osp.join(sample_path, 'x.npy'))
    x = x[:m, :]
    x = x.astype(float)
    _, k = x.shape
    print(x, x.shape, x.dtype, '\n')

    # get current observable values
    df = pd.read_csv(osp.join(sample_path, 'data_out', 'observables.traj'), delimiter="\t", header=None)
    df = df.dropna(axis=1)
    df = df.drop(df.columns[0] ,axis=1)
    lam = df.mean().values
    print('obj measured: ', lam)

    # sorens method
    obj_goal = []
    for i in range(k):
        # seqi = x[:, i]
        seqi = np.loadtxt(osp.join(sample_path, f"seq{i}.txt"))[:m]
        print(f'\ni={i}', seqi)
        for j in range(k):
            if j < i:
                # don't double count
                continue
            # seqj = x[:, j]
            seqj = np.loadtxt(osp.join(sample_path, f"seq{i}.txt"))[:m]
            print(f'\tj={j}', seqj)
            result = seqi @ y @ seqj
            # outer = np.outer(seqi, seqj)
            # result = np.sum(outer * y)

            result /= m**2 # average
            result /= y_max # convert to probability
            obj_goal.append(result)
    print(np.round(obj_goal, 7))
    print(obj_goal / lam)

if __name__ == '__main__':
    main()
