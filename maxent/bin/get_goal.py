import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

'''TICG-MaxEnt algorithm. Calculates goal observables from simulation data
this is for testing maxent on data generated from simulations,
typically goal observables will actually be calculated from experimental contact maps'''

obs_file = sys.argv[1]
out_file = sys.argv[2]

df = pd.read_csv(obs_file, delimiter="\t", header=None)
df = df.drop(df.columns[0], axis=1)
df = df.dropna(axis=1)
goal = df.mean().values
#goal /= sum(goal)

obj_goal = open(out_file, "w+")

for x in goal:
	obj_goal.write(str(x) + " ")

obj_goal.write("\n")

'''
df = pd.read_csv("data_out/observables.traj", delimiter="\t", header=None)
df = df.drop(df.columns[0], axis=1)
df = df.dropna(axis=1)
goal = df.mean().values
#goal /= sum(goal)

obj_goal = open("obj_goal.txt" , "w+")

for x in goal:
	obj_goal.write(str(x) + " ")

obj_goal.write("\n")
'''
