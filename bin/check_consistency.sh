#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=1024
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed=12
seqSeed=13
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
nSweeps=10000
dumpFrequency=1000
dumpStatsFrequency=100
trackContactMap='false'
TICGSeed=10
dense='true'
diagBins=32
minDiagchi=2
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

# i=1000
# useE='false'
# useS='false'
# useD='false'
# run &
#
# i=1001
# useE='false'
# useS='true'
# useD='false'
# run &
#
# i=1002
# useE='true'
# useS='false'
# useD='false'
# run &
#
# i=1003
# useE='false'
# useS='true'
# useD='true'
# run &
#
# i=1004
# useE='true'
# useS='false'
# useD='true'
# run &

sample=1462
baseDataFolder="/home/erschultz/dataset_11_18_22/samples/sample${sample}"
chiMethod="none"
m=1024

i=1005
seqMethod="${baseDataFolder}/sd.npy-S"
chiDiagMethod="none"
run &

i=1006
seqMethod="${baseDataFolder}/sd_wrong.npy-S"
chiDiagMethod="none"
run &

wait
