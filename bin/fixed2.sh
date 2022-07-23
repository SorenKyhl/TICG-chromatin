#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
k=4
m=1024
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
startSample=1
relabel='none'
diag='false'
nSweeps=500000
lmbda=0.8
maxDiagChi=5
chiSeed='31'
seqSeed='31'
chiMethod='random'
minChi=-0.4
maxChi=0.4
fillDiag='none'
overwrite=1
dumpFrequency=10000

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

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

k=0
nSweeps=1000
dumpFrequency=100000
TICGSeed=10
chiDiagMethod='linear'
chiDiagSlope=2000
dense='true'
denseCutoff=0.125


i=12
m=512
maxDiagChi=20
diagBins=16
run &

# i=164
# m=8192
# parallel='true'
# numThreads=6
# run

wait
