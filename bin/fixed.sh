#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=512
dataFolder="/home/erschultz/dataset_test_diag1024"
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
nSweeps=2000000
dumpFrequency=100000
TICGSeed=10
diag='true'
diagBins=32

chiDiagMethod='log'

m=512

m=1024
diagBins=16
i=101
chiDiagSlope=5120
maxDiagChi=2
for i in $( seq 101 120 )
do
	echo $i $maxDiagChi $chiDiagSlope
	run &


	maxDiagChi=$(( $maxDiagChi + 2 ))
	if ! (( $i % 10 ))
	then
		echo "wait"
		wait
		chiDiagSlope=$(( $chiDiagSlope * 2 ))
		maxDiagChi=2
	fi
	i=$(( $i + 1 ))

done

wait
