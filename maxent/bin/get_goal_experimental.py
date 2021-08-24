import numpy as np
import matplotlib.pyplot as plt
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
    m = 1024
    offset = 0
    y = np.load('../sequences_to_contact_maps/dataset_04_18_21/samples/sample40/y.npy')[:m, :m]
    # y = np.array([[1, 0, 0],[1, 1, 0], [1,1,1]])
    y_max = np.max(y)
    print('y_max: ', y_max)
    # y = np.triu(y, k = offset) # avoid double counting due to symmetry
    y = y.astype(float)

    print(y, y.shape, y.dtype, '\n')

    x = np.load('../sequences_to_contact_maps/dataset_04_18_21/samples/sample40/x.npy')
    x = x[:m, :]
    x = x.astype(float)
    print(x, x.shape, x.dtype, '\n')

    # x = np.array([[0,1], [1,0], [1,1]])

    # my method
    t0 = time.time()
    obj_goal = []
    for i in range(2):
        seqi = x[:, i]
        print('\ni={}'.format(i), seqi)
        for j in range(2):
            if j < i:
                # don't double count
                continue
            seqj = x[:, j]
            print('\tj={}'.format(j), seqj)
            left = seqi @ y
            result = seqi @ y @ seqj
            result /= m**2 # take average
            result /= y_max # convert to probability
            obj_goal.append(result)
    t = round(time.time() - t0, 5)
    print('time: ', t)
    print(obj_goal, '\n')

    # sorens method
    t0 = time.time()
    obj_goal = []
    for i in range(2):
        seqi = x[:, i]
        print('\ni={}'.format(i), seqi)
        for j in range(2):
            if j < i:
                # don't double count
                continue
            seqj = x[:, j]
            print('\tj={}'.format(j), seqj)
            outer = np.outer(seqi, seqj)
            result = np.mean(np.multiply(outer, y))
            result /= y_max # convert to probability
            obj_goal.append(result)
    t = round(time.time() - t0, 5)
    print('time: ', t)
    print(obj_goal)

if __name__ == '__main__':
    main()
