#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=512
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed=12
seqSeed=11
chiMethod="random"
seqMethod="random"
minChi=-0.4
maxChi=0.4
fillDiag='none'
overwrite=1

source activate python3.8_pytorch1.8.1

cd ~/TICG-chromatin

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

k=4
nSweeps=100
dumpFrequency=1
dumpStatsFrequency=1
trackContactMap='false'
TICGSeed=12
dense='true'
diagBins=32
minDiagchi=0
maxDiagChi=10
nSmallBins=16
smallBinSize=4
bigBinSize=-1
nBigBins=-1
bondLength=28

# this checks consistency of different energy formalisms
# all seeds are fixed
gridMoveOn='true'
trackContactMap='false'
chiDiagMethod="linear"
updateContactsDistance='false'
m=512

i=1000
useL='false'
useS='false'
useD='false'
run &

i=1001
useL='true'
useS='false'
useD='false'
run &

i=1002
useL='false'
useS='false'
useD='true'
run &

i=1003
useL='true'
useS='true'
useD='true'
run &


wait
