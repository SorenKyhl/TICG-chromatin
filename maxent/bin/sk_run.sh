#!/bin/bash

today=$(date +'%m_%d_%y')
scratchDir=${1:-'/scratch/midway2/erschultz/TICG_maxent'}
mode=${2:-"both"}
gamma=${3:-1}
trust_region=${4:-10000}
equilib_sweeps=${5:-10000}
production_sweeps=${6:-50000}
num_iterations=${7:-100}
goal_specified=${8:-0}
overwrite=${9:-0}
method=${10:-"n"}

echo "running maxent with:"
echo "dir:"
echo $scratchDir
echo "mode:"
echo $mode
echo "gamma:"
echo $gamma
echo "trust_region:"
echo $trust_region
echo "equilib_sweeps"
echo $equilib_sweeps
echo "production_sweeps"
echo $production_sweeps
echo "num_iterations"
echo $num_iterations
echo "goal_specified"
echo $goal_specified
echo "overwrite"
echo $overwrite
echo "method"
echo $method

# move to scratch
if ! [[ -d $scratchDir ]]
then
	echo "scratchDir does not exist"
	mkdir $scratchDir
	#exit 1
fi
cp -r resources $scratchDir
cd $scratchDir

if ! [[ -d resources ]]
then
	echo "resources does not exist"
	exit 1
fi

# other parameters
configFileName='config.json'
saveFileName='equilibrated.xyz'
proj_bin="/home/skyhl/Documents/TICG-chromatin/maxent/bin" # location of algorithm scripts
nchis=$(head -1 "resources/chis.txt" | wc -w)
k=$(jq .nspecies "resources/${configFileName}")
ndiagchis=$(head -1 "resources/chis_diag.txt" | wc -w)

# set up other files
cp resources/chis.txt .
cp resources/chis_diag.txt .
touch track.log

run_simulation () {
	STARTTIME=$(date +%s)
	# resources must be in working directory
	mkdir "iteration$it"
	cp resources/* "iteration$it"
	if [ $mode == "plaid" ];
	then
		python3 $proj_bin/update_chis.py --it $it --k $k
	elif [ $mode == "diag" ];
	then
		python3 $proj_bin/update_diag.py --it $it
	elif [ $mode == "both" ];
	then
		python3 $proj_bin/update_chis.py --it $it --k $k
		python3 $proj_bin/update_diag.py --it $it
	fi
	cd "iteration${it}"

	# equilibrate system
	python3 $proj_bin/jsed.py $configFileName nSweeps $equilib_sweeps i
	~/Documents/TICG-chromatin/src/TICG-engine > equilib.log
	$proj_bin/fork_last_snapshot.sh $saveFileName
	mv data_out equilib_out

	# set up production run
	python3 $proj_bin/jsed.py $configFileName load_configuration_filename $saveFileName s
	python3 $proj_bin/jsed.py $configFileName nSweeps $production_sweeps i
	python3 $proj_bin/jsed.py $configFileName seed $RANDOM i
	~/Documents/TICG-chromatin/src/TICG-engine > production.log

	if [ $it -gt $(($num_iterations - 1)) ]
	then
		python3 ~/Documents/TICG-chromatin/scripts/contact_map.py --save_npy
	fi
	mv data_out production_out
	cd $scratchDir

	ENDTIME=$(date +%s)
	echo "finished iteration ${it}: $(($ENDTIME - $STARTTIME)) seconds"
}

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
if [ $num_iterations -gt 0 ]
then
	for it in $(seq 1 $(($num_iterations)))
	do
		run_simulation
		# update chis via newton's method
		python3 $proj_bin/newton_step.py $it $gamma $mode $goal_specified $trust_region $method >> track.log
		# update plots
		python3 $proj_bin/plot_convergence.py --mode $mode --k $k
		python3 $proj_bin/contactmap.py $it
	done
fi

# run longer simulation
it=$(($num_iterations + 1))
python3 $proj_bin/jsed.py "resources/${configFileName}" dump_frequency 50000 i
production_sweeps=500000
run_simulation
python3 $proj_bin/contactmap.py $it

