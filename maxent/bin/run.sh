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
# |- bin
#    |- algorithm scripts
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

# change to max ent directory
cd ~/TICG-chromatin/maxent

# command-line arguments
today=$(date +'%m_%d_%y')
outputDir=${1:-"/project2/depablo/erschultz/maxent_${today}"}
gamma=${2:-0.00001}
gamma_diag=${3:-0.00001}
mode=${4:-"plaid"}
production_sweeps=${5:-50000}
equilib_sweeps=${6:-10000}
goal_specified=${7:-0}
num_iterations=${8:-50}

# other parameters
scratchDir='/scratch/midway2/erschultz/TICG_maxent'
configFileName='config.json'
saveFileName='equilibrated.xyz'
proj_root=$(pwd)
proj_bin="$(pwd)/bin"                # location of algorithm scripts
nchis=$(head -1 "resources/chis.txt" | wc -w)
k=$(jq .nspecies "resources/${configFileName}")
ndiagchis=$(head -1 "resources/chis_diag.txt" | wc -w)

# directory checks
if [ -d $outputDir ]
then
	# don't overrite previous results!
	echo "output directory already exists"
	exit 1
fi

if [[ -d resources && -d bin ]]
then
	echo "resources and bin exist"
else
	echo "resources or bin directory do not exist"
	exit 1
fi

run_simulation () {
	# resources must be in working directory
	mkdir "iteration$it"
	cp resources/* "iteration$it"
	if [ $mode == "plaid" ];
	then
		python3 $proj_bin/update_chis.py --it $it --k $k
	elif [ $mode == "diag" ];
	then
		python3 $proj_bin/update_diag.py $it
	elif [ $mode == "both" ];
	then
		python3 $proj_bin/update_chis.py --it $it --k $k
		python3 $proj_bin/update_diag.py --it $it
	fi
	cd "iteration${it}"

	# equilibrate system
	python3 $proj_bin/jsed.py $configFileName nSweeps $equilib_sweeps i
	~/TICG-chromatin/TICG-engine > equilib.log
	$proj_bin/fork_last_snapshot.sh $saveFileName
	mv data_out equilib_out

	# set up production run
	python3 $proj_bin/jsed.py $configFileName load_configuration_filename $saveFileName s
	python3 $proj_bin/jsed.py $configFileName nSweeps $production_sweeps i
	python3 $proj_bin/jsed.py $configFileName seed $RANDOM i
	~/TICG-chromatin/TICG-engine > production.log

	python3 ~/TICG-chromatin/scripts/contact_map.py --save_npy
	mv data_out production_out

	echo "finished iteration $it"
	cd $scratchDir
}

# set up scratch and output directory
mkdir -p $scratchDir
mkdir -p $outputDir
cp -r resources $scratchDir
cd $scratchDir
mv resources/chis.txt .
mv resources/chis_diag.txt .
touch track.log

# iteration 0
it=0
if [ $goal_specified -eq 1 ]
then
	# if goal is specified, just move in goal files and do not simulate
	mv resources/obj_goal.txt .
	mv resources/obj_goal_diag.txt .
else
	# if goal is not specified, simulate iteration 0 and calculate the goals from that simulation
	run_simulation
fi

# maxent optimization
for it in $(seq 1 $(($num_iterations)))
do
	STARTTIME=$(date +%s)
	run_simulation
	ENDTIME=$(date +%s)
	echo "iteration ${it} time: $(($ENDTIME - $STARTTIME)) seconds"

	# update chis via newton's method
	STARTTIME=$(date +%s)
	python3 $proj_bin/newton_step.py $it $gamma $gamma_diag $mode $goal_specified >> track.log
	ENDTIME=$(date +%s)
	echo "newton time: $(($ENDTIME - $STARTTIME)) seconds"

	# update plots
	STARTTIME=$(date +%s)
	gnuplot $proj_bin/plot.p $nchis $ndiagchis
	ENDTIME=$(date +%s)
	echo "plot time: $(($ENDTIME - $STARTTIME)) seconds"
	gnuplot $proj_bin/plot.p $nchis $ndiagchis
done

# run longer simulation
STARTTIME=$(date +%s)
it=$(($num_iterations + 1))
production_sweeps=500000
run_simulation
ENDTIME=$(date +%s)
echo "long simulation time: $(($ENDTIME - $STARTTIME)) seconds"

# move data to output directory
mv $scratchDir/* $outputDir
