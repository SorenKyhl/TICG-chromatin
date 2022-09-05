#!/bin/bash

today=$(date +'%m_%d_%y')
outputDir='/scratch/midway2/erschultz/TICG_maxent'
mode="both" # c
gamma=1
trust_region=10000 # t
equilib_sweeps=10000 # e
production_sweeps=50000 # s
num_iterations=100 #  n
goal_specified=1 # g
overwrite=0 # o
method="n" # m
parallel_cores=0
randomize_seed=1 # r

show_help()
{
	echo "-h help"  
	echo "-o outputDir"
	echo "-c [ both | plaid | diag ]"
	echo "-g gamma"
	echo "-t trust_region"
	echo "-e equilib_sweeps"
	echo "-s production_sweeps"
	echo "-n num_iterations"
	echo "-x goal_specified"
	echo "-w overwrite"
	echo "-m [ n | g ]"
	echo "-p parallel_cores"
}


while getopts "hxwo:c:g:t:e:s:n:m:p:r:" opt; do
	case $opt in
		h) show_help ;;
		o) outputDir=$(pwd)/$OPTARG ;;
		c) mode=$OPTARG ;;
		g) gamma=$OPTARG ;;
		t) trust_region=$OPTARG ;;
		e) equilib_sweeps=$OPTARG ;;
		s) production_sweeps=$OPTARG ;;
		n) num_iterations=$OPTARG ;;
		x) goal_specified=1 ;;
		w) overwrite=1 ;;
		m) method=$OPTARG ;;
		p) parallel_cores=$OPTARG ;;
		r) randomize_seed=$OPTARG ;;
	esac
done

echo "running maxent with:"
echo "dir:"
echo $outputDir
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
echo "parallel"
echo $parallel_cores
echo "randomize_seed"
echo $randomize_seed


get_rng ()
{
    if [[ $randomize_seed -eq 1 ]]
    then
        echo $RANDOM
    else
        echo 1
    fi
}

# move to scratch
if ! [[ -d $outputDir ]]
then
	echo "outputDir does not exist"
	mkdir $outputDir
	#exit 1
fi

cp -r resources/ $outputDir
cd $outputDir

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
	cp $(find resources/ ! -name "experimental_hic.npy") "iteration$it"
	ln -s "resources/expermental_hic.npy" "iteration$it"
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
	python3 $proj_bin/jsed.py $configFileName load_configuration true b
	python3 $proj_bin/jsed.py $configFileName load_configuration_filename $saveFileName s
	python3 $proj_bin/jsed.py $configFileName nSweeps $production_sweeps i
	python3 $proj_bin/jsed.py $configFileName seed $(get_rng) i

	if [ "$parallel_cores" -eq 0 ]
	then
		~/Documents/TICG-chromatin/src/TICG-engine > production.log
	else
		run_parallel
	fi

	if [ $it -gt $(($num_iterations - 1)) ]
	then
		python3 ~/Documents/TICG-chromatin/scripts/contact_map.py --save_npy
	fi
	mv data_out production_out
	cd $outputDir

	ENDTIME=$(date +%s)
	echo "finished simulating ${it}: $(($ENDTIME - $STARTTIME)) seconds"
}

run_parallel()
{
	for (( i=1; i<=$parallel_cores; i++ ))
	do
		python3 $proj_bin/jsed.py $configFileName seed $(get_rng) i
		./TICG-engine "core$i" > "core$i.log" &
	done
	wait

	mkdir production_out
	cp core1/* production_out
	cat core*/energy.traj > production_out/energy.traj
	cat core*/observables.traj > production_out/observables.traj
	cat core*/diag_observables.traj > production_out/diag_observables.traj
	cat core*/output.xyz > production_out/output.xyz

	python3 $proj_bin/cat_contacts.py core*/contacts.txt production_out/contacts.txt
}

# iteration 0
it=0
if [ $goal_specified -eq 1 ]
then
	# if goal is specified, just move in goal files and do not simulate
	cp resources/obj_goal.txt .
	cp resources/obj_goal_diag.txt .
else
	# if goal is not specified, simulate iteration 0 and calculate the goals from that simulation
	run_simulation
fi

# maxent optimization
if [ $num_iterations -gt 0 ]
then
	for it in $(seq 1 $(($num_iterations)))
	do
		STARTTIME=$(date +%s)
		run_simulation
		# update chis via newton's method
		python3 $proj_bin/newton_step.py $it $gamma $mode $goal_specified $trust_region $method >> track.log
		# update plots
		python3 $proj_bin/plot_convergence.py --mode $mode --k $k
		python3 $proj_bin/contactmap.py $it
		(cd iteration$it && python3 $proj_bin/analysis.py)
		ENDTIME=$(date +%s)
		echo "finished iteration ${it}: $(($ENDTIME - $STARTTIME)) seconds"
	done
fi

# run longer simulation
it=$(($num_iterations + 1))
python3 $proj_bin/jsed.py "resources/${configFileName}" dump_frequency 50000 i
production_sweeps=500000
run_simulation
python3 $proj_bin/contactmap.py $it
(cd iteration$it && python3 $proj_bin/analysis.py)

