#!/bin/bash

today=$(date +'%m_%d_%y')
scratchDir='/scratch/midway2/erschultz/TICG_maxent'
mode="both" # c
gamma=1
trust_region=10000 # t
equilib_sweeps=10000 # e
production_sweeps=50000 # p
num_iterations=100 #  n
goal_specified=1 # g
overwrite=0 # o
method="n" # m

show_help()
{
	echo "-h help"  
	echo "-s scratchDir"
	echo "-c [ both | plaid | diag ]"
	echo "-g gamma"
	echo "-t trust_region"
	echo "-e equilib_sweeps"
	echo "-p production_sweeps"
	echo "-n num_iterations"
	echo "-x goal_specified"
	echo "-o overwrite"
	echo "-m [ n | g ]"
}


while getopts "hxos:c:g:t:e:p:n:m:" opt; do
	case $opt in
		h) show_help ;;
		s) scratchDir=$(pwd)/$OPTARG ;;
		c) mode=$OPTARG ;;
		g) gamma=$OPTARG ;;
		t) trust_region=$OPTARG ;;
		e) equilib_sweeps=$OPTARG ;;
		p) production_sweeps=$OPTARG ;;
		n) num_iterations=$OPTARG ;;
		x) goal_specified=1 ;;
		o) overwrite=1 ;;
		m) method=$OPTARG ;;
	esac
done

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
		mv diff*.png iteration$it
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

