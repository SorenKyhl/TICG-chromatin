import os.path as osp
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def step(parameter_file, obs_file, convergence_file, goal_file, gamma, it, goal_specified):

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
    print("current chi values: ", current_chis)

    # get current observable values
    df = pd.read_csv(osp.join("iteration{}".format(it), "production_out", obs_file), delimiter="\t", header=None)
    df = df.dropna(axis=1)
    df = df.drop(df.columns[0] ,axis=1)
    lam = df.mean().values


    # normalizing factor (June 16 2020)
    # damping coefficients have to be really small to prevent overshoot.
    # is there a scaling factor that can adjust for this?....
    #Ngrid = 17**3
    #nucl_per_bead = 10

    #lam = lam * Ngrid * nucl_per_bead**2
    #obj_goal = obj_goal * Ngrid * nucl_per_bead**2
    #df = df * Ngrid * nucl_per_bead**2

    current_chis = np.array(current_chis)

    difference = obj_goal - lam
    B = df.cov().values
    Binv = np.linalg.pinv(B)
    step = Binv@difference
    print('step: ', step)
    print('lam: ', lam)

    new_chis = current_chis - gamma*step

    print("new chi values: ", new_chis)

    f_chis = open(parameter_file, "a")
    np.savetxt(f_chis, new_chis, newline=" ", fmt="%.5f")
    f_chis.write("\n")
    f_chis.close()

    #howfar = sum(abs(difference))/sum(obj_goal)
    howfar = np.sqrt(difference@difference)/np.sqrt(obj_goal@obj_goal)

    with open(convergence_file, "a") as f:
        f.write(str(howfar) + '\n')

def copy_chis(parameter_file, obs_file, convergence_file, goal_file, gamma, it, goal_specified = None):
    ''' for parameters that are not optimized, just copy chis to next iteration'''
    # load current chi parameters
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

    with open(convergence_file, "a") as f:
        f.write(str(0) + '\n')

def main():
    '''
    Calculates one step of the TICG-MaxEnt optimization routine.
    i.e. updates chis based on observables from simulation
    '''

    it = int(sys.argv[1])                    # iteration number
    gamma_plaid  = float(sys.argv[2])        # damping coefficient
    gamma_diag = float(sys.argv[3])          # dampling coefficient
    mode = sys.argv[4]                       # plaid, diag, or both
    goal_specified = str2bool(sys.argv[5])   # if true, will read from obj_goal.txt and obj_goal_diag.txt.
                                             # if false, will calculate goals from iteration1 observables

    print(goal_specified)

    print('Iteration Number {}'.format(it))
    print("gamma_plaid: {}".format(gamma_plaid))
    print("gamma_diag: {}".format(gamma_diag))

    if mode == "diag":
        diag_fn = step
        fn = copy_chis
    elif mode == "plaid":
        diag_fn = copy_chis
        fn = step
    elif mode == "both":
        diag_fn = step
        fn = step
    else:
        raise Exception("Unknown mode: {}".format(mode))

    parameter_file = "chis_diag.txt"
    obs_file = "diag_observables.traj"
    convergence_file = "convergence_diag.txt"
    goal_file = "obj_goal_diag.txt"
    gamma = gamma_diag
    diag_fn(parameter_file, obs_file, convergence_file, goal_file, gamma, it, goal_specified)

    parameter_file = "chis.txt"
    obs_file = "observables.traj"
    convergence_file = "convergence.txt"
    goal_file = "obj_goal.txt"
    gamma = gamma_plaid
    fn(parameter_file, obs_file, convergence_file, goal_file, gamma, it, goal_specified)
    print('\n')

if __name__ == '__main__':
    main()
