#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

chi='polynomial'
param_setup
k=4
m=1500
dataFolder="/home/erschultz/dataset_test3"
scratchDir='/home/erschultz/scratch'
startSample=1
relabel='none'
diag='false'
nSweeps=500000
lmbda=0.8
maxDiagChi=5
chiSeed='none'
chi='none'
minChi=-2
maxChi=1
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


method='random'
nSweeps=1000000
diag='true'
maxDiagChi=2
for i in $( seq 11 20 )
do
	echo "i=${i}, maxDiagChi=${maxDiagChi}"
	run &
	maxDiagChi=$(( $maxDiagChi + 2 ))
done

wait
