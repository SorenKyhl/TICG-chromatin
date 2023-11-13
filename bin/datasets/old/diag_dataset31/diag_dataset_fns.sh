#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
dataset=dataset_03_22_23
dataFolder="/project2/depablo/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'
#
# dataFolder="/home/erschultz/${dataset}"
# scratchDir='/home/erschultz/scratch'

m=512
overwrite=1
useL='true'
useE='true'
useD='true'

nSweeps=500000
dumpFrequency=100000
TICGSeed=10
dense='false'
diagBins=512
phiChromatin=0.06
gridSize=24
beadVol=520
bondLength=16.5
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
