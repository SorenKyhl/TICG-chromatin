import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

'''calculates one step of the TICG-MaxEnt optimization routine
i.e. updates chis based on observables from simulation
'''

it = int(sys.argv[1])               # iteration number
gamma_plaid  = float(sys.argv[2])   # damping coefficient
gamma_diag = float(sys.argv[3])     # dampling coefficient
mode = sys.argv[4]                  # plaid, diag, or both
goal_specified = int(sys.argv[5])   # if true, will read from obj_goal.txt and obj_goal_diag.txt. 
                                    # if false, will calculate goals from iteration1 observables

print("iteration number " + str(it) + " gamma_plaid : " + str(gamma_plaid))
print("iteration number " + str(it) + " gamma_diag : " + str(gamma_diag))

def step(parameter_file, obs_file, convergence_file, goal_file, gamma):

    if goal_specified:
        print("READING FROM OBJ_GOAL")
        f_obj_goal = open(goal_file, "r")
        obj_goal = f_obj_goal.readline().split()
        obj_goal = np.array([float(x) for x in obj_goal])
        f_obj_goal.close()
    else:
        # get goal observables from first iteration
        print("READING FROM OBS")
        df = pd.read_csv("iteration" + str(0) + "/data_out/" + obs_file, delimiter="\t", header=None)
        df = df.dropna(axis=1)
        df = df.drop(df.columns[0] ,axis=1)
        obj_goal = df.mean().values

    print("obj goal:", obj_goal)

    # read in current chis
    f_chis = open(parameter_file, "r")
    lines = f_chis.readlines()
    current_chis = lines[it].split()
    current_chis = [float(x) for x in current_chis]
    f_chis.close()
    print("current chi values: ", current_chis)

    # get current observable values
    df = pd.read_csv("iteration" + str(it) + "/data_out/" + obs_file, delimiter="\t", header=None)
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
    Binv = np.linalg.inv(B)
    step = Binv@difference
    print(it)
    print(step)
    print(obj_goal)
    print(lam)

    new_chis = current_chis - gamma*step

    print("new chi values: ", new_chis)

    f_chis = open(parameter_file, "a")
    np.savetxt(f_chis, new_chis, newline=" ", fmt="%.5f") 
    f_chis.write("\n")
    f_chis.close()

    #howfar = sum(abs(difference))/sum(obj_goal)
    howfar = np.sqrt(difference@difference)/np.sqrt(obj_goal@obj_goal)

    convergence = open(convergence_file, "a")
    convergence.write(str(howfar) + '\n')
    convergence.close()

def copy_chis(parameter_file, obs_file, convergence_file, goal_file, gamma):
    ''' for parameters that are not optimized, just copy chis to next iteration'''
    # load current chi parameters
    f_chis = open(parameter_file, "r")
    lines = f_chis.readlines()
    current_chis = lines[it].split()
    current_chis = [float(x) for x in current_chis]
    f_chis.close()

    new_chis = current_chis

    #write chi parameters to next iteration, unchanged
    f_chis = open(parameter_file, "a")
    np.savetxt(f_chis, new_chis, newline=" ", fmt="%.5f") 
    f_chis.write("\n")
    f_chis.close()
    

if (mode == "diag"):
    parameter_file = "chis_diag.txt"
    obs_file = "diag_observables.traj"
    convergence_file = "convergence_diag.txt"
    goal_file = "obj_goal_diag.txt"
    gamma = gamma_diag
    step(parameter_file, obs_file, convergence_file, goal_file, gamma)

    parameter_file = "chis.txt"
    obs_file = "observables.traj"
    convergence_file = "convergence.txt"
    goal_file = "obj_goal.txt"
    gamma = gamma_plaid
    copy_chis(parameter_file, obs_file, convergence_file, goal_file, gamma)

if (mode == "plaid"):
    parameter_file = "chis_diag.txt"
    obs_file = "diag_observables.traj"
    convergence_file = "convergence_diag.txt"
    goal_file = "obj_goal_diag.txt"
    gamma = gamma_diag
    copy_chis(parameter_file, obs_file, convergence_file, goal_file,  gamma)

    parameter_file = "chis.txt"
    obs_file = "observables.traj"
    convergence_file = "convergence.txt"
    goal_file = "obj_goal.txt"
    gamma = gamma_plaid
    step(parameter_file, obs_file, convergence_file, goal_file,  gamma)

if (mode == "both"):
    parameter_file = "chis_diag.txt"
    obs_file = "diag_observables.traj"
    convergence_file = "convergence_diag.txt"
    goal_file = "obj_goal_diag.txt"
    gamma = gamma_diag
    step(parameter_file, obs_file, convergence_file,goal_file,  gamma)

    parameter_file = "chis.txt"
    obs_file = "observables.traj"
    convergence_file = "convergence.txt"
    goal_file = "obj_goal.txt"
    gamma = gamma_plaid
    step(parameter_file, obs_file, convergence_file,goal_file,  gamma)



