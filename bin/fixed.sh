#! /bin/bash
#SBATCH --job-name=TICG_fixed
#SBATCH --output=TICG_fixed.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000

method='ground_truth'
sample=40
sampleFolder="/project2/depablo/erschultz/dataset_04_18_21/samples/sample$sample"
saveFileName='equilibrated.xyz'
m=1024
k=2
startSimulation=11
numSimulations=11
chi="-1&1\\1&0"
overwrite=0

outputFolder="/project2/depablo/erschultz/dataset_fixed"
scratchDir='/scratch/midway2/erschultz/TICG_fixed'

# get inputxyz
cd "/project2/depablo/skyhl/dataset_04_18_21/samples/sample$sample"
~/TICG-chromatin/maxent/bin/fork_last_snapshot.sh "${scratchDir}/${saveFileName}"

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
	python3 ~/TICG-chromatin/scripts/get_config.py --save_chi --chi=$chi --m $m --k $k --ensure_distinguishable --load_configuration_filename $saveFileName --nSweeps 5000000 --dump_frequency 500000 > log.log

	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --sample_folder $sampleFolder --sample $sample --m $m --k $k --save_npy >> log.log

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
	mv config.json data_out log.log x.npy y.npy y.png chis.txt chis.npy $dir
	for i in $(seq 0 $(($k-1)))
	do
		mv seq${i}.txt $dir
	done
done

# clean up
rm default_config.json input1024.xyz
