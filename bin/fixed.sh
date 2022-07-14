#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=512
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed='31'
seqSeed='31'
chiMethod='random'
minChi=-0.4
maxChi=0.4
fillDiag='none'
overwrite=1

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

# cd ~/TICG-chromatin/srcs
# make
# mv TICG-engine ..

k=0
nSweeps=10000
dumpFrequency=100
TICGSeed=10
diag='true'
diagBins=32
maxDiagChi=10
chiDiagMethod='linear'
m=512

i=5
constantChi=0
run &

i=6
constantChi=2
run &

i=7
constantChi=0
k=1
chi=2
seqMethod='block-A512'
run &


wait
