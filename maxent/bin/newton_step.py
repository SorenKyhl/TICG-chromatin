import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def step(parameter_file, obs_file, convergence_file, goal_file, gamma, it, goal_specified, trust_region, method):

    # get goals
    if goal_specified:
        print("READING FROM OBJ_GOAL")
        with open(goal_file, "r") as f_obj_goal:
            obj_goal = f_obj_goal.readline().split()
            obj_goal = np.array([float(x) for x in obj_goal])
    else:
        # get goal observables from zeroth iteration
        print("READING FROM OBS")
        df = pd.read_csv(osp.join("iteration{}" .format(0), "production_out", obs_file), delimiter="\t", header=None)
        df = df.dropna(axis=1)
        df = df.drop(df.columns[0] ,axis=1)
        obj_goal = df.mean().values

    print("obj goal: ", obj_goal)

    # read in current chis
    with open(parameter_file, "r") as f_chis:
        lines = f_chis.readlines()
        current_chis = lines[it].split()
        current_chis = [float(x) for x in current_chis]
        current_chis = np.array(current_chis)
    print("current chi values: ", current_chis)

    # get current observable values
    df = pd.read_csv(osp.join("iteration{}".format(it), "production_out", obs_file), delimiter="\t", header=None)
    df = df.dropna(axis=1)
    df = df.drop(df.columns[0] ,axis=1)
    lam = df.mean().values
    B = df.cov().values

    # vbead = 520
    # vcell = 28.7**3
    # B /= vcell/vbead

    new_chis, howfar = newton(lam, obj_goal, B,  gamma, current_chis, trust_region, method)

    f_chis = open(parameter_file, "a")
    np.savetxt(f_chis, new_chis, newline=" ", fmt="%.5f")
    f_chis.write("\n")
    f_chis.close()

    with open(convergence_file, "a") as f:
        f.write(str(howfar) + '\n')

def both_step(parameter_files, obs_files, convergence_files, goal_files, gamma, it, goal_specified, trust_region, method):

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
    #print("obj goal: ", obj_goal)

    # read in current chis
    nchis = [None, None]
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
    it_root = osp.join("iteration{}".format(it), "production_out")

    df_total = pd.DataFrame()
    for f in obs_files:
        df = pd.read_csv(osp.join(it_root, f), delimiter="\t", header=None)
        df = df.dropna(axis=1)
        df = df.drop(df.columns[0] ,axis=1)
        df_total= pd.concat((df_total, df), axis=1)

    #df_total /= np.max(obj_goal)
    #obj_goal /= np.max(obj_goal)

    lam = df_total.mean().values
    B = df_total.cov().values

    # vbead = 520
    # vcell = 28.7**3
    # B /= vcell/vbead

    print("obj goal: ", obj_goal)
    print("lam: ", lam)

    new_chis, howfar = newton(lam, obj_goal, B, gamma, current_chis, trust_region, method)


    index = 0;
    for i, f in enumerate(parameter_files):
        f_chis = open(f, "a")
        np.savetxt(f_chis, new_chis[index:index+nchis[i]], newline=" ", fmt="%.5f")
        f_chis.write("\n")
        f_chis.close()

        index += nchis[i]

    for f in convergence_files:
        with open(f, "a") as f:
            f.write(str(howfar) + '\n')

def newton(lam, obj_goal, B, gamma, current_chis, trust_region, method):
    difference = obj_goal - lam
    Binv = np.linalg.pinv(B)
    if method == "n":
        step = Binv@difference
    elif method == "g":
        step = difference

    steplength = np.sqrt(step@step)

    print("========= step before gamma: ", steplength)
    print('step: ', step)
    print('lam: ', lam)
    print('difference: ', difference)
    print('B: ', B)

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

    plt.plot(-1*difference)
    plt.savefig("-1*difference.png")

    new_chis = current_chis - step
    print(f"new chi values: {new_chis}\n")

    howfar = np.sqrt(difference@difference)/np.sqrt(obj_goal@obj_goal)

    return new_chis, howfar

def copy_chis(parameter_file, obs_file, convergence_file, goal_file, gamma, it, goal_specified = None, trust_region = None, method = None):
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

    with open(convergence_file, "a") as f:
        f.write(str(0) + '\n')

def main():
    '''
    Calculates one step of the TICG-MaxEnt optimization routine.
    i.e. updates chis based on observables from simulation
    '''

    it = int(sys.argv[1])                    # iteration number
    gamma = float(sys.argv[2])               # damping coefficient
    mode = sys.argv[3]                       # plaid, diag, or both
    goal_specified = str2bool(sys.argv[4])   # if true, will read from obj_goal.txt and obj_goal_diag.txt.
                                             # if false, will calculate goals from iteration1 observables
    trust_region = float(sys.argv[5])
    method = str(sys.argv[6])

    print(f'Goal specified: {goal_specified}')
    print(f'Iteration Number {it}')
    print(f"gamma: {gamma}")
    print(f"method: {method}")

    if mode == "both":
        parameter_files = ["chis.txt", "chis_diag.txt"]
        obs_files = ["observables.traj", "diag_observables.traj"]
        convergence_files = ["convergence.txt", "convergence_diag.txt"]
        goal_files = ["obj_goal.txt", "obj_goal_diag.txt"]
        both_step(parameter_files, obs_files, convergence_files, goal_files, gamma, it, goal_specified, trust_region, method)
    else:
        if mode == "diag":
            diag_fn = step
            fn = copy_chis
        elif mode == "plaid":
            diag_fn = copy_chis
            fn = step

        parameter_file = "chis_diag.txt"
        obs_file = "diag_observables.traj"
        convergence_file = "convergence_diag.txt"
        goal_file = "obj_goal_diag.txt"
        diag_fn(parameter_file, obs_file, convergence_file, goal_file, gamma, it, goal_specified, trust_region, method)

        parameter_file = "chis.txt"
        obs_file = "observables.traj"
        convergence_file = "convergence.txt"
        goal_file = "obj_goal.txt"
        fn(parameter_file, obs_file, convergence_file, goal_file, gamma, it, goal_specified, trust_region, method)

if __name__ == '__main__':
    main()
