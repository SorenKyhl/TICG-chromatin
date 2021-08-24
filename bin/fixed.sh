#! /bin/bash
#SBATCH --job-name=TICG_fixed
#SBATCH --output=TICG_fixed.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000

method='ground_truth'
m=1024
k=2
startSimulation=13
numSimulations=13
overwrite=0

outputFolder="/project2/depablo/erschultz/dataset_fixed"
scratchDir='/scratch/midway2/erschultz/TICG_fixed'

# move utils to scratch
mkdir -p $scratchDir
cd ~/TICG-chromatin/utils
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
	~/TICG-chromatin/TICG-engine >> log.log

  # calculate contact map
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --save_npy

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
