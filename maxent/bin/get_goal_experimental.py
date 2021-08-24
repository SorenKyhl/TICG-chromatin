import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import csv
import time

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--m', type=int, default=1024, help='number of particles, will crop seq{i}.txt and y')
    parser.add_argument('--k', type=int, help='number of particle types (inferred from chi if None)')
    parser.add_argument('--contact_map', type=str, help='filepath to contact map')
    parser.add_argument('--verbose', action='store_true', help='true for verbose mode')
    parser.add_argument('--triu', action='store_true', help='true to use triu')

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

    y = y.astype(float) # ensure float
    y = y[:args.m, :args.m] # crop to m
    y_max = np.max(y)
    if args.triu:
        y = np.triu(y) # avoid double counting due to symmetry
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
            result /= args.m**2 # take average
            result /= y_max # convert from freq to prob
            obj_goal.append(result)

    if args.verbose:
        print('obj_goal: ', obj_goal)

    with open('obj_goal.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(obj_goal)
    with open('obj_goal_diag.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(np.zeros(20))

def test():
    sample_path='../compare_eric/sample13'

    m = 1024
    offset = 0
    # y = np.load(osp.join(sample_path, 'y.npy'))[:m, :m]
    y = np.loadtxt(osp.join(sample_path, 'data_out', 'contacts.txt'))[:m, :m]
    y_max = np.max(y)
    print('y_max: ', y_max)
    # y = np.triu(y, k = offset) # avoid double counting due to symmetry
    y = y.astype(float)
    # y /= y_max # convert to probability
    # np.fill_diagonal(y, 0)

    print(y, y.shape, y.dtype, '\n')

    # x = np.load(osp.join(sample_path, 'x.npy'))
    # x = x[:m, :]
    # x = x.astype(float)
    # _, k = x.shape
    # print(x, x.shape, x.dtype, '\n')
    k = 2

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
        seqi = np.loadtxt(osp.join(sample_path, "seq{}.txt".format(i)))[:m]
        print('\ni={}'.format(i), seqi)
        for j in range(k):
            if j < i:
                # don't double count
                continue
            # seqj = x[:, j]
            seqj = np.loadtxt(osp.join(sample_path, "seq{}.txt".format(j)))[:m]
            print('\tj={}'.format(j), seqj)
            outer = np.outer(seqi, seqj)
            result = np.sum(outer * y)
            result /= m**2 # average
            result /= y_max # convert to probability
            obj_goal.append(result)
    print(np.round(obj_goal, 7))
    print(obj_goal / lam)

if __name__ == '__main__':
    test()
