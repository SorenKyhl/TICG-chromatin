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
numSimulations=2
# chi="1&2&-1&1.5\\2&1&-1&-0.5\\-1&-1&1&1.5\\1.5&-0.5&1.5&1"
chi='none'
fillOffdiag=0


today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_${today}"
scratchDir='/scratch/midway2/erschultz/TICG'

# directory checks
if [ -d $dataFolder ]
then
	# don't overrite previous results!
	echo "output directory already exists"
	exit 1
fi

# move utils to scratch
mkdir -p $scratchDir
cd ~/TICG-chromatin/utils
cp input1024.xyz "${scratchDir}/input1024.xyz"
cp default_config.json "${scratchDir}/default_config.json"

# change directory to scratch
cd $scratchDir

# activate python
source activate python3.8_pytorch1.8.1_cuda10.2

for i in $(seq 2 $numSimulations)
do
  # set up config.json
	python3 ~/TICG-chromatin/scripts/get_config.py --save_chi --chi $chi --m $m --fill_offdiag $fillOffdiag > log.log

	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --p_switch $pSwitch --k $k

	# run simulation
	~/TICG-chromatin/TICG-engine >> log.log

  # calculate contact map
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m

	# move inputs and outputs to own folder
	dir="${dataFolder}/samples/sample${i}"
	mkdir -p $dir
	mv config.json data_out log.log x.npy y.npy y.png chis.txt chis.npy $dir
	for i in $(seq 0 $(($k-1)))
	do
		mv seq${i}.txt $dir
	done
done
