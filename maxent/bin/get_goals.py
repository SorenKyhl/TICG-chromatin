import epilib as ep
import numpy as np
import sys


bins = int(sys.argv[1])
sim = ep.Sim(".")
diag = ep.get_goal_diag(sim.gthic, bins, dense_diagonal_on=True)
np.savetxt("obj_goal_diag.txt", diag, newline=" ", fmt="%.8f")
