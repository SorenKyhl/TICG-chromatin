#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=TICG.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

method='random'
m=1024
pSwitch=0.05
k=4
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_${today}"
numSimulations=1
chi="1&2&-1&1.5\\2&1&-1&-0.5\\-1&-1&1&1.5\\1.5&-0.5&1.5&1"

# move utils to scratch
scratchDir='/scratch/midway2/erschultz/TICG'
mkdir -p $scratchDir
cd ~/TICG-chromatin/utils
cp input1024.xyz "${scratchDir}/input1024.xyz"
cp default_config.json "${scratchDir}/default_config.json"

# change cwd to scratch
cd $scratchDir

# activate python
source activate python3.8_pytorch1.8.1_cuda10.2

for i in $(seq 1 $numSimulations)
do
	python3 ~/TICG-chromatin/scripts/get_config.py --save_chi --chi $chi --m $m > log.log

	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --p_switch $pSwitch --k $k

	# run simulation
	~/TICG-chromatin/TICG-engine >> log.log

  # calculate contact map
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m

	# move output to own folder
	dir="${dataFolder}/samples/sample${i}"
	mkdir -p $dir
	mv data_out log.log x.npy y.npy y.png chis.txt chis.npy $dir
	for i in $(seq 0 $(($k-1)))
	do
		mv seq${i}.txt $dir
	done
done

mv config.json ${dataFolder}
