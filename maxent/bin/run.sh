#!/bin/bash
today=$(date +'%m_%d_%y')

# default args
outputDir="/project2/depablo/erschultz/maxent_${today}" # o
scratchDir='/scratch/midway2/erschultz/TICG_maxent' # d
mode="both" # c
gamma=1 # g
trust_region=50 # t
equilib_sweeps=10000 # e
production_sweeps=50000 # s
start_iteration=1
num_iterations=100 #  n
goal_specified=0 # z
overwrite=0 # w
method="n" # m
parallel_cores=0 # p
randomize_seed=0 # r
final_sim_production_sweeps=1000000 # f

show_help()
{
	echo "-h help"
	echo "-o outputDir"
	echo "-d scratchDir"
	echo "-c [ both | plaid | diag ]"
	echo "-g gamma"
	echo "-t trust_region"
	echo "-e equilib_sweeps"
	echo "-s production_sweeps"
	echo "-q start_iteration"
	echo "-n num_iterations"
	echo "-z goal_specified"
	echo "-w overwrite"
	echo "-m [ n | g ]"
	echo "-p parallel_cores"
	exit 1
}

while getopts "h:o:c:g:t:e:s:n:m:p:r:d:q:f:z:w:" opt; do
	case $opt in
		h) show_help ;;
		o) outputDir=$OPTARG ;;
		d) scratchDir=$OPTARG ;;
		c) mode=$OPTARG ;;
		g) gamma=$OPTARG ;;
		t) trust_region=$OPTARG ;;
		e) equilib_sweeps=$OPTARG ;;
		s) production_sweeps=$OPTARG ;;
		q) start_iteration=$OPTARG ;;
		n) num_iterations=$OPTARG ;;
		z) goal_specified=$OPTARG ;;
		w) overwrite=$OPTARG ;;
		m) method=$OPTARG ;;
		p) parallel_cores=$OPTARG ;;
		r) randomize_seed=$OPTARG ;;
		f) final_sim_production_sweeps=$OPTARG
	esac
done

echo $@
echo "running maxent with:"
echo "dir:"
echo $outputDir
echo "scratchDir:"
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
echo "parallel"
echo $parallel_cores
echo "randomize_seed"
echo $randomize_seed
echo "final_sim_production_sweeps"
echo $final_sim_production_sweeps

get_rng ()
{
    if [[ $randomize_seed -eq 1 ]]
    then
        echo $RANDOM
    else
        echo 1
    fi
}

# cd to scratch
if ! [[ -d $scratchDir ]]
then
	echo "scratchDir does not exist ${scratchDir}"
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
	cd $scratchDir
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
		python3 $proj_bin/update_params.py --it $it --mode $mode
	fi
	cd "iteration${it}"

	if [ $num_iterations -gt 0 ]
	then
		# equilibrate system
		# no need to do so if num_iterations==0
		python3 $proj_bin/jsed.py $configFileName nSweeps $equilib_sweeps i
		python3 $proj_bin/jsed.py $configFileName seed $(get_rng) i
		~/TICG-chromatin/TICG-engine > equilib.log
		$proj_bin/fork_last_snapshot.sh $saveFileName
		tar -czf equilib_out.tar.gz data_out
		rm -r data_out
	fi

	# production
	python3 $proj_bin/jsed.py $configFileName nSweeps $production_sweeps i
	if [ $num_iterations -gt 0 ]
	then
		# only change seed if num_iterations > 0 (allows for easier reproducibility tests)
		python3 $proj_bin/jsed.py $configFileName seed $(get_rng) i
		# equilib will only exist if num_iterations > 0
		python3 $proj_bin/jsed.py $configFileName load_configuration_filename $saveFileName s
		python3 $proj_bin/jsed.py $configFileName load_configuration true b

	fi
	~/TICG-chromatin/TICG-engine > production.log
	mv data_out production_out

	# delete files copied from resources to save space
	rm *.txt *.npy

	ENDTIME=$(date +%s)
	echo "finished iteration ${it}: $(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes ($(( $ENDTIME - $STARTTIME )) seconds)"

	cd $scratchDir
}

directory checks
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
	mv resources/chi* .
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
if [ $goal_specified -eq 1 ]
then
	# if goal is specified, just move in goal files and do not simulate
	mv resources/obj_goal* .
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
		python3 $proj_bin/newton_step.py --it $it --gamma $gamma --mode $mode --goal_specified $goal_specified --trust_region $trust_region >> track.log

		# convert to tarball
		cd "iteration${it}"
		tar -czf production_out.tar.gz production_out
		rm -r production_out
		cd $scratchDir

		# update plots
		python3 $proj_bin/plot_convergence.py --mode $mode > plot.log
	done
fi

# run longer simulation
it=$(($num_iterations + 1))
python3 $proj_bin/jsed.py "resources/${configFileName}" dump_frequency 50000 i
production_sweeps=$final_sim_production_sweeps
run_simulation
cd "iteration${it}"
python3 $proj_bin/analysis.py > analysis.log
OVERALLENDTIME=$(date +%s)
echo "finished entire simulation: $(( $(( $OVERALLENDTIME - $OVERALLSTARTTIME )) / 60 )) minutes ($(( $OVERALLENDTIME - $OVERALLSTARTTIME )) seconds)"


# move data to output directory
mv $scratchDir/* $outputDir
rm -d $scratchDir
