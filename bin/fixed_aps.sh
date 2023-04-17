#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh
source activate python3.9_pytorch1.9

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

param_setup
m=250
dataset=dataset_test
baseDataFolder="/home/erschultz/${dataset}"
dataFolder=$baseDataFolder
scratchDir='/home/erschultz/scratch'
overwrite=1

k=4
nSweeps=5000000
dumpFrequency=10000
TICGSeed=10
maxChi=2
minChi=-2
chiSeed=3
seqSeed=4
diag='true'
dense='false'
diagBins=250
bondLength=16.5
gridSize=24
useD='true'
useL='true'
useE='true'


sample=4001
chiDiagMethod="linear"
maxDiagChi=2
chiMethod="random"
seqMethod="block-A30-B50-A40-C60-B30-D40"
i="${sample}"
# run &

sample=4002
m=500
diagBins=500
maxDiagChi=1
seqMethod="block-A60-B100-A80-C120-B60-D80"
i="${sample}"
run &

sample=4003
m=500
diagBins=500
maxDiagChi=2
seqMethod="block-A60-B100-A80-C120-B60-D80"
i="${sample}"
run &

sample=4004
m=500
diag='false'
i="${sample}"
run &


wait
