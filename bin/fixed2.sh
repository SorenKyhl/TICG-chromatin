#! /bin/bash

method='ground_truth'
m=1024
k=2
startSimulation=14
numSimulations=14
overwrite=0
pwd=$(pwd)
outputFolder="C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps/dataset_fixed"
scratchDir=$outputFolder

# move utils to scratch
mkdir -p $scratchDir
cd utils
cp input1024.xyz "${scratchDir}/input1024.xyz"
cp default_config.json "${scratchDir}/default_config.json"

# change directory to scratch
cd $scratchDir

# activate python
source activate python3.8_pytorch1.8.1_cuda10.2

for i in $(seq $startSimulation $numSimulations)
do
  # set up config.json
	cp ~/TICG-chromatin/utils/config_soren.json .

	# generate sequences
	cp ~/TICG-chromatin/utils/seq0.txt .
  cp ~/TICG-chromatin/utils/seq1.txt .

	# run simulation
	$pwd/TICG-engine >> log.log

  # calculate contact map
  python3 $pwd/scripts/contact_map.py --m $m --save_npy

	# move inputs and outputs to own folder
	dir="${outputFolder}/samples/sample${i}"
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
	mkdir -p $dir
	mv config_soren.json data_out log.log x.npy y.npy y.png chis.txt chis.npy $dir
	for i in $(seq 0 $(($k-1)))
	do
		mv seq${i}.txt $dir
	done
done

# clean up
rm default_config.json input1024.xyz
