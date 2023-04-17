#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
# dataFolder="/project2/depablo/erschultz/dataset_09_30_22"
# scratchDir='/home/erschultz/scratch-midway2'
#
dataFolder="/home/erschultz/dataset_9_30_22"
scratchDir='/home/erschultz/scratch'

k=4
m=1024
relabel='none'
lmbda='none'
pSwitch=0.05
chiSeed='none'
seqSeed='none'
chiMethod='random'
minChi=-2
maxChi=2
fillDiag='none'
overwrite=1
useS='true'
useD='true'

nSweeps=1000000
dumpFrequency=100000
TICGSeed=10
chiDiagMethod='logistic'
dense='false'
diagBins=1024
nSmallBins=16
smallBinSize=4
nBigBins=-1
bigBinSize=-1
diagCutoff='none'
phiChromatin=0.06
diagStart=0
bondLength=20
trackContactMap='true'


run()  {
	# move utils to scratch
	scratchDirI="${scratchDir}/${i}"
	move

	check_dir

	random_inner

	# clean up
	rm -f default_config.json *.xyz
	rm -d $scratchDirI
}
