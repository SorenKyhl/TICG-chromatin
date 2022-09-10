import argparse
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--it', type=int,
                        help='current iteration')
    parser.add_argument('--gamma', type=float,
                        help='damping coefficient')
    parser.add_argument('--mode', type=str,
                        help='{plaid, diag, both, all}')
    parser.add_argument('--goal_specified', type=str2bool,
                        help='True to read from obj_goal.txt'
                            'False to calculate goals from iteration 1')
    parser.add_argument('--trust_region', type=float,
                        help='newton step trust region')
    parser.add_argument('--min_diag_chi', type=str2float,
                        help='min value of diag chi during newton step')
    parser.add_argument('--method', type=str, default='n',
                        help='method for newtons method')

    args, _ = parser.parse_known_args()

    # for compatibility with Soren
    if args.it is None and args.mode is None:
        # assume the rest are None as well
        args.it = int(sys.argv[1])                    # iteration number
        args.gamma = float(sys.argv[2])               # damping coefficient
        args.mode = sys.argv[3]                       # plaid, diag, both, or all
        args.goal_specified = str2bool(sys.argv[4])   # if true, will read from obj_goal.txt and obj_goal_diag.txt.
                                                      # if false, will calculate goals from iteration1 observables
        args.trust_region = float(sys.argv[5])
        args.method = sys.argv[6]

    return args

def str2float(v):
    """
    Helper function for argparser, converts str to float if possible.

    Inputs:
        v: string
    """
    if v is None:
        return v
    elif isinstance(v, str):
        if v.lower() == 'none':
            return None
        elif v.replace('.', '').replace('-', '').isnumeric():
            return float(v)
        else:
            raise argparse.ArgumentTypeError('none or float expected not {}'.format(v))
    else:
        raise argparse.ArgumentTypeError('String value expected.')

def str2bool(v):
    """
    Helper function for argparser, converts str to boolean for various string inputs.
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Inputs:
        v: string
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def step(parameter_files, obs_files, convergence_file, goal_files, gamma, it,
        goal_specified, trust_region, min_val, method):

    # get goals
    if goal_specified:
        print("READING FROM OBJ_GOAL")
        obj_goal = np.hstack([np.loadtxt(f) for f in goal_files])
    else:
        # get goal observables from zeroth iteration
        print("READING FROM OBS")
        obs = pd.DataFrame()
        for f in obs_files:
            df = pd.read_csv(f, sep='\t', header=None)
            df = df.drop(df.columns[0], axis=1)
            obs = pd.concat((obs, df), axis=1)

        obj_goal = obs.mean().values

    # read in current chis
    nchis = [None]*len(parameter_files)
    current_chis = []
    for i, f in enumerate(parameter_files):
        with open(f, "r") as f_chis:
            lines = f_chis.readlines()
            f_current_chis = lines[it].split()
            f_current_chis = [float(x) for x in f_current_chis]
            nchis[i] = len(f_current_chis)
            current_chis.extend(f_current_chis)
    current_chis = np.array(current_chis)
    print("current chi values: ", current_chis)

    # get current observable values
    it_root = osp.join(f"iteration{it}", "production_out")

    df_total = pd.DataFrame()
    for f in obs_files:
        try:
            # arr = np.loadtxt(f)[:, 1:]
            df = pd.read_csv(osp.join(it_root, f), delimiter="\t", header=None)
            df = df.dropna(axis=1)
            df = df.drop(df.columns[0] ,axis=1)
            df_total = pd.concat((df_total, df), axis=1)
        except:
            print('Error with ', osp.join(it_root, f))
            raise

    lam = df_total.mean().values
    B = df_total.cov().values

    print("obj goal: ", obj_goal)

    new_chis, howfar = newton(lam, obj_goal, B, gamma, current_chis, trust_region, method)


    index = 0;
    for i, f in enumerate(parameter_files):
        f_chis = open(f, "a")
        new_chis_i = new_chis[index:index+nchis[i]]
        if 'diag' in f and min_val is not None:
            new_chis_i[new_chis_i < min_val] = min_val
        np.savetxt(f_chis, new_chis_i, newline=" ", fmt="%.5f")
        f_chis.write("\n")
        f_chis.close()

        index += nchis[i]

    with open(convergence_file, "a") as f:
        f.write(str(howfar) + '\n')

def newton(lam, obj_goal, B, gamma, current_chis, trust_region, method):
    difference = obj_goal - lam
    Binv = np.linalg.pinv(B)
    if method == "n":
        step = Binv@difference
    elif method == "g":
        step = difference
    else:
        raise Exception(f'Unrecognized method: {method}')

    steplength = np.sqrt(step@step)

    print("========= step before gamma: ", steplength)
    print('lam: ', lam)
    print('difference: ', difference)
    print('step: ', step)
    # print('B: ', B)

    if gamma != 1:
        step *= gamma
        steplength = np.sqrt(step@step)
        print("========= step after gamma: ", steplength)
        print('step: ', step)

    if steplength > trust_region:
        step /= steplength
        step *= trust_region
        steplength = np.sqrt(step@step)
        print("======= OUTSIDE TRUST REGION =========")
        print("========= steplength: ", steplength)
        print("========= trust_region: ", trust_region)
        print('step: ', step)
        print('lam: ', lam)

    new_chis = current_chis - step
    print(f"new chi values: {new_chis}\n")

    howfar = np.sqrt(difference@difference)/np.sqrt(obj_goal@obj_goal)

    return new_chis, howfar

def copy_chis(parameter_file, it):
    ''' for parameters that are not optimized, just copy chis to next iteration'''
    # load current chi parameters
    if osp.exists(parameter_file):
        with open(parameter_file, "r") as f_chis:
            lines = f_chis.readlines()
            current_chis = lines[it].split()
            current_chis = [float(x) for x in current_chis]

        new_chis = current_chis

        #write chi parameters to next iteration, unchanged
        f_chis = open(parameter_file, "a")
        np.savetxt(f_chis, new_chis, newline=" ", fmt="%.5f")
        f_chis.write("\n")
        f_chis.close()
    else:
        pass
        # assume that this is ok

def main():
    '''
    Calculates one step of the TICG-MaxEnt optimization routine.
    i.e. updates chis based on observables from simulation
    '''
    args = getArgs()
    if args.it == 1:
        print(args)

    print(f'Iteration Number {args.it}')

    if args.mode == 'all':
        parameter_files = ["chis.txt", "chis_diag.txt", "chi_constant.txt"]
        obs_files = ["observables.traj", "diag_observables.traj", "constant_observable.traj"]
        goal_files = ["obj_goal.txt", "obj_goal_diag.txt", "obj_goal_constant.txt"]
    elif args.mode == "both":
        parameter_files = ["chis.txt", "chis_diag.txt"]
        obs_files = ["observables.traj", "diag_observables.traj"]
        goal_files = ["obj_goal.txt", "obj_goal_diag.txt"]
    elif args.mode == "diag":
        parameter_files = ["chis_diag.txt"]
        obs_files = ["diag_observables.traj"]
        goal_files = ["obj_goal_diag.txt"]
        copy_chis("chis.txt", args.it)
    elif args.mode == "plaid":
        parameter_files = ["chis.txt"]
        obs_files = ["observables.traj"]
        goal_files = ["obj_goal.txt"]
        copy_chis("chis_diag.txt", args.it)

    step(parameter_files, obs_files, 'convergence.txt', goal_files,
                args.gamma, args.it, args.goal_specified, args.trust_region,
                args.min_diag_chi, args.method)

if __name__ == '__main__':
    main()
