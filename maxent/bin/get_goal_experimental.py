import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--k', type=int, help='number of particle types (inferred from chi if None)')
    parser.add_argument('--contact_map', type=str, help='filepath to contact map')

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

    y = y[:args.m, :args.m] # crop to m
    y = np.triu(y) # avoid double counting due to symmetry

    for i in range(args.k):
        seqi = np.loadtxt("seq{}.txt".format(i))
        for j in range(args.k):
            if j < i:
                # don't double count
                continue
            seqj = np.loadtxt("seq{}.txt".format(i))
            result = seqi @ y @ seqj
            result /= args.m # take average
            result /= y_max # convert from freq to prob

    obj_goal.append(result)

    np.savetxt('obj_goal.txt', np.array(obj_goal))
    np.savetxt('obj_goal_diag.txt', np.zeros(20))


if __name__ == '__main__':
    main()
