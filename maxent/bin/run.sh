#!/bin/bash

# TICG-MaxEnt optimization algorithm.
#
# This script will execute maximum entropy optimization of chi parameters for the TICG-chromatin model of genome architecture
#
# TODO: explain necessary file structure
#
# ----------- command-line arguments --------------
# outputDir: directory where all simulations and results are output
#
# gamma: newton's method relaxation constant for plaid chi optimization. often needs to be << 1, otherwise overshoot.
#
# gamma_diag: newton's method relaxation constant for diagonal chi optimization. often needs to be << 1, otherwise overshoot.
#
# mode:
		# plaid: optimize only plaid chis, with fixed diagonal chis
		# diag: optimize only diagonal chis, with fixed plaid chis
		# both: optimize both plaid and diagonal chis
#
# production_sweeps: number of production sweeps
#
# equilib_sweeps: number of equilibration sweeps
#
# goal_specified:
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
# num_iterations: number of iterations of Newton's method
	# set to 0 to only perform final production run
#
# overwrite: True to overwrite existing results in outputDir
#
# scratchDir: scratch directory for temporary file storage
#
# final_sim_production_sweeps: Number of sweeps in final production run (want this to be much larger than production_sweeps)
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
today=$(date +'%m_%d_%y')
outputDir=${1:-"/project2/depablo/erschultz/maxent_${today}"}
gamma=${2:-1}
trust_region=${3:-10}
minDiagChi=${4:-"none"}
mode=${5:-"plaid"}
production_sweeps=${6:-50000}
equilib_sweeps=${7:-10000}
goal_specified=${8:-"false"}
start_iteration=${9:-1}
num_iterations=${10:-50}
overwrite=${11:-0}
scratchDir=${12:-'/scratch/midway2/erschultz/TICG_maxent'}
final_sim_production_sweeps=${13:-1000000}

echo $@

# cd to scratch
if ! [[ -d $scratchDir ]]
then
	echo "scratchDir does not exist"
	exit 1
fi
cd $scratchDir

# other parameters
module load jq
configFileName='config.json'
saveFileName='equilibrated.xyz'
proj_bin=~/TICG-chromatin/maxent/bin # location of algorithm scripts
if [ -f "resources/chis.txt" ]
then
	nchis=$(head -1 "resources/chis.txt" | wc -w)
fi
if [ -f "resources/${configFileName}" ]
then
	k=$(jq .nspecies "resources/${configFileName}")
fi
if [ -f "resources/chis_diag.txt" ]
then
	ndiagchis=$(head -1 "resources/chis_diag.txt" | wc -w)
fi

run_simulation () {
	STARTTIME=$(date +%s)
	# resources must be in working directory
	itDir="iteration$it"
	if [ -d $itDir ]
	then
		echo "removing ${itDir}"
		rm -r $itDir
	fi
	mkdir $itDir
	cp resources/* $itDir
	if [ $num_iterations -gt 0 ]
	then
		# no need to update if num_iterations==0
		if [ $mode == "plaid" ];
		then
			python3 $proj_bin/update_chis.py --it $it
		elif [ $mode == "diag" ];
		then
			python3 $proj_bin/update_diag.py --it $it
		elif [ $mode == "both" ];
		then
			python3 $proj_bin/update_chis.py --it $it
			python3 $proj_bin/update_diag.py --it $it
		fi
	fi
	cd "iteration${it}"

	if [ $num_iterations -gt 0 ]
	then
		# equilibrate system
		# no need to do so if num_iterations==0
		python3 $proj_bin/jsed.py $configFileName nSweeps $equilib_sweeps i
		~/TICG-chromatin/TICG-engine > equilib.log
		$proj_bin/fork_last_snapshot.sh $saveFileName
		mv data_out equilib_out
		python3 $proj_bin/jsed.py $configFileName load_configuration_filename $saveFileName s
	fi

	# production
	python3 $proj_bin/jsed.py $configFileName nSweeps $production_sweeps i
	if [ $num_iterations -gt 0 ]
	then
		# don't change seed if num_iterations==0 (allows for reproducibility)
		python3 $proj_bin/jsed.py $configFileName seed $RANDOM i
	fi
	~/TICG-chromatin/TICG-engine > production.log

	mv data_out production_out
	cd $scratchDir

	ENDTIME=$(date +%s)
	echo "finished iteration ${it}: $(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes ($(( $ENDTIME - $STARTTIME )) seconds)"

}

# directory checks
if [ -d $outputDir ]
then
	if [ $overwrite -eq 1 ]
	then
		echo "output directory already exists - overwriting"
		rm -r $outputDir
	else
		# don't overrite previous results!
		echo "output directory already exists - aborting"
		exit 1
	fi
fi

if ! [[ -d resources ]]
then
	echo "resources does not exist"
	exit 1
fi

mkdir -p $outputDir
if [ $start_iteration -le 1 ]
then
	mv resources/chis* .
	mv resources/*.png .
	mv resources/*.log .
	touch track.log
fi

if [ $mode = 'none' ]
then
	num_iterations=0
	goal_specified='false'
fi

# iteration 0
it=0
if [ $goal_specified = "true" ]
then
	# if goal is specified, just move in goal files and do not simulate
	mv resources/obj_goal.txt .
	mv resources/obj_goal_diag.txt .
elif [ $num_iterations -gt 0 ]
then
	# if goal is not specified, simulate iteration 0 and calculate the goals from that simulation
	run_simulation
fi

# maxent optimization
OVERALLSTARTTIME=$(date +%s)
if [ $num_iterations -gt 0 ]
then
	for it in $(seq $start_iteration $num_iterations)
	do
		run_simulation
		# update chis via newton's method
		python3 $proj_bin/newton_step.py --it $it --gamma $gamma --mode $mode --goal_specified $goal_specified --trust_region $trust_region --min_diag_chi $minDiagChi >> track.log

		# update plots
		python3 $proj_bin/plot_convergence.py --mode $mode
	done
fi

# run longer simulation
it=$(($num_iterations + 1))
python3 $proj_bin/jsed.py "resources/${configFileName}" dump_frequency 50000 i
production_sweeps=$final_sim_production_sweeps
run_simulation
OVERALLENDTIME=$(date +%s)
echo "finished entire simulation: $(( $(( $OVERALLENDTIME - $OVERALLSTARTTIME )) / 60 )) minutes ($(( $OVERALLENDTIME - $OVERALLSTARTTIME )) seconds)"


# move data to output directory
mv $scratchDir/* $outputDir
rm -d $scratchDir
