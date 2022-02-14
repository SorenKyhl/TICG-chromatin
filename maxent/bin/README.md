#!/bin/bash


# TICG-MaxEnt optimization algorithm.
#
# This script will execute maximum entropy optimization of chi parameters for the TICG-chromatin model of genome architecture
# usage: (while in proj_root/)
#	./bin/run.sh
#
# Required folder structure:
# proj_root
# |- resources
#    |- config.json
#    |- init_config.xyz
#    |- seqs.txt
#    |- TICG-engine
#    |- chis.txt
#    |- chis_diag.txt
#    |- obj_goal.txt (optional)
#    |- obj_goal_diag.txt (optional)
#
# the maximum entropy procedure requires several files in the resources folder:
# config.json:
#	specifies most of the simulation paramters that are constant throughout maximum entropy optimization.
#	the only exceptions are the number of equilibration and production sweeps, which are command-line arguments to this script,
#	... and the chi values, which are updated by after each maximum entropy step by the helper functions "update_chis.sh" and
#	"update_diagonal.py"
#
# init_config.xyz
#	initial configuration for all simulations TODO: randomize this initial configuration. It's not essential, the initial
#	configuration doesn't affect the output after equilibration, and the seed for each simulation is random
#
# seqs.txt
#	one-dimensinoal vectors corresponding to epigenetic bead labels
#
# TICG-engine
#	binary executable, simulation engine
#
# chis.txt
#	space-separated list of chi values in alphbetical order (chiAA, chiAB, chiAC ... chiBB, chiBC ... chiCC).
#	each line records the chi values for that iteration (i.e. line 0 has chi values for iteration 0)
#	the initial chi values to start maximum entropy must be specified in the first line of this file.
#
# chis_diag.txt
#	identical to chis.txt, except for diagonal chis
#
# obj_goal.txt (optional)
#	space-separated list of goal observables corresponding to each chi parameter. (optional: if $goal_specified
#   is false, goal observables will be calculated from iteration 0 simulations)
#
# obj_gial_diag.txt (optional)
#	identical to obj_goal_txt, except for diagonal chis
#
# ----------- command-line arguments --------------
# outputDir
#	directory where all simulations and results are output
#
# gamma
#	newton's method relaxation constant for plaid chi optimization. often needs to be << 1, otherwise overshoot.
#
# gamma_diag
#	newton's method relaxation constant for diagonal chi optimization. often needs to be << 1, otherwise overshoot.
#
# mode
#	plaid: optimize only plaid chis, with fixed diagonal chis
#	diag: optimize only diagonal chis, with fixed plaid chis
#	both: optimize both plaid and diagonal chis
#
# production_sweeps
#	number of production sweeps
#
# equilib_sweeps
#	number of equilibration sweeps
#
# goal_specified
#	if false, the user does not need to specify observables. instead, the user should specify the first two lines (zeroth and first)
#   in chis.txt. The first line (zeroth simulation) are the chi parameters for which the goal observables will be calculated.
#	the second line (first simulation) are the inital chi parameters to start the maximum entropy procdeure,
#	which will subsequently attempt to match the conditions observed in iteration 0. The same is true of chis_diag.txt
#
#	if true, the user must specifiy goal observables in the resources/obj_goal.txt and resources/obj_goal_diag.txt
#	maximum entropy optimization will begin with iteration 1, and iterate until an optimum is reached
#	initial values for chi paramters must be specified in chis.txt and chis_diag.txt.
#	NOTE: since this version does not require iteration0, the algorithm IGNORES THE FIRST LINE (corresponding to iteration 0) of chis.txt
#   Therefore, the first two lines of chis.txt should be identical, and the second line (corresponding to iteratino1)
#	defines the initial chi parameters for the simulation. The same is true of chis_diag.txt
#
#
# ------------ results -----------
# in outputDir:
#	iteration<i>
#		directories that contain simulation results from each iteration, including config file and other inputs
#
#	chis.txt, chis_diag.txt
#		space-separated list of chi parameters, each line corresponds to a new iteration. the first line is always iteration 0
#
#	pchis.png, pchi

# command-line arguments
