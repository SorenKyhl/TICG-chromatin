import numpy as np

def adjust(filename):
    x = np.loadtxt(filename)
    vbead = 520
    vcell = 28.7**3
    x *= vbead/vcell
    np.savetxt(filename, x, fmt="%.8f", newline=" ")
    return

fnames = ["obj_goal.txt", "obj_goal_diag.txt"]

for f in fnames:
    adjust(f)
