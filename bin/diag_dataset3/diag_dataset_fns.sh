#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
dataset=dataset_12_05_22
dataFolder="/project2/depablo/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch-midway2'

# dataFolder="/home/erschultz/${dataset}"
# scratchDir='/home/erschultz/scratch'

m=1024
overwrite=1
useE='true'
useD='true'

nSweeps=500000
dumpFrequency=100000
TICGSeed=10
dense='false'
diagBins=1024
phiChromatin=0.06
beadVol=520
gridSize=28.7
bondLength=28
maxDiagChi='none'
trackContactMap='true'


run()  {
	# move utils to scratch
	scratchDirI="${scratchDir}/${i}"
	move

	check_dir

	argsFile="${dataFolder}/setup/sample_${i}.txt"

	random_inner

	# clean up
	rm -f default_config.json *.xyz
	rm -d $scratchDirI
}
