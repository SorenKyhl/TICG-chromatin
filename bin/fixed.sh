#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=512
dataFolder="/home/erschultz/dataset_test_diag512"
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
nSweeps=1000000
dumpFrequency=10000
diag='true'
maxDiagChi=2
chiDiagMethod='log'
chiDiagSlope=10
for i in $( seq 1 100 )
do
	echo "i=${i}, maxDiagChi=${maxDiagChi}, chiDiagSlope=${chiDiagSlope}"
	run &

	if [ $( expr $i % 10 ) -eq 0 ]
	then
		maxDiagChi=2
		chiDiagSlope=$(( $chiDiagSlope * 2 ))
	fi
	maxDiagChi=$(( $maxDiagChi + 2 ))

	if [ $( expr $i % 15 ) -eq 0 ]
	then
		wait
	fi
done


wait
