#! /bin/bash

k=$2
chi=$3
m=$4
startSimulation=$5
numSimulations=$6
dataFolder=$7
relabel=$8
diag=$9
local=${10}

# other params
method='random'
pSwitch=0.05
maxDiagChi=0.1
overwrite=1
nSweeps=500000
dumpFrequency=50000

# below does nothing if chi is given
minChi=0
maxChi=2
fillDiag=-1

echo $@

echo $local
if [ $local -eq 0 ]
then
	echo scratch
	scratchDir="/scratch/midway2/erschultz/TICG${1}"

	# move utils to scratch
	mkdir -p $scratchDir
	cd ~/TICG-chromatin/utils
	cp input1024.xyz "${scratchDir}/input1024.xyz"
	cp default_config.json "${scratchDir}/default_config.json"

	cd $scratchDir
else
	echo data
	# move utils
	mkdir -p $dataFolder
	cd ~/TICG-chromatin/utils
	cp input1024.xyz "${dataFolder}/input1024.xyz"
	cp default_config.json "${dataFolder}/default_config.json"

	cd $dataFolder
fi

# activate python
echo $local
if [ $local -eq 0 ]
then
	echo activate
	source activate python3.8_pytorch1.8.1_cuda10.2
else
  source activate python3.8_pytorch1.8.1_cuda11.1
fi

for i in $(seq $startSimulation $numSimulations)
do
	echo $i
	dir="${dataFolder}/samples/sample${i}"
	# directory checks
	if [ -d $dir ]
	then
		if [ $overwrite -eq 1 ]
		then
			echo "output directory already exists - overwriting"
			rm -r $dir
		else
			# don't overrite previous results!
			echo "output directory already exists - aborting"
			exit 1
		fi
	fi

  # set up config.json
	python3 ~/TICG-chromatin/scripts/get_config.py --save_chi --chi=$chi --m $m --k $k --min_chi $minChi --max_chi $maxChi --fill_diag $fillDiag --ensure_distinguishable --diag $diag --max_diag_chi $maxDiagChi --n_sweeps $nSweeps --dump_frequency $dumpFrequency > log.log

	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --p_switch $pSwitch --k $k --save_npy --relabel $relabel >> log.log

	# run simulation
	~/TICG-chromatin/TICG-engine >> log.log

  # calculate contact map
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --save_npy

	# move inputs and outputs to own folder
	mkdir -p $dir
	mv config.json data_out log.log *.npy *.png chis.txt $dir
	for i in $(seq 0 $(($k-1)))
	do
		mv seq${i}.txt $dir
	done
done

# clean up
rm default_config.json input1024.xyz
