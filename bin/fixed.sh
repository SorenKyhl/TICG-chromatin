#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=512
dataFolder="/home/erschultz/dataset_test_diag"
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
nSweeps=500000
dumpFrequency=20000
diag='true'
diagBins=40
maxDiagChi=10
chiDiagMethod='linear'
m=2048
i=140
run &

i=141
m=1024
diagBins=20
maxDiagChi=5
chiDiagMethod='linear'
run &


wait
