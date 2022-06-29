#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
k=4
m=1024
dataFolder="/home/erschultz/dataset_test2"
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

nSweeps=1000000
diag='true'
maxDiagChi=2
chiMultiplier=1
for i in $( seq 1 16 )
do
	echo "i=${i}, maxDiagChi=${maxDiagChi}, chiMultiplier=${chiMultiplier}"
	run &
	if [ $i -lt 5 ]
	then
		maxDiagChi=$(( $maxDiagChi + 3 ))
	elif [ $i -gt 3 ] && [ $i -lt 9 ]
	then
	 chiMultiplier=$(($chiMultiplier * 2 ))
 elif [ $i -gt 7 ] && [ $i -lt 13 ]
	then
		maxDiagChi=$(( $maxDiagChi - 3 ))
	elif [ $i -gt 12 ]
	then
		chiMultiplier=$(($chiMultiplier / 2 ))
	fi

	# if [ $( expr $i % 10 ) -eq 0 ]
	# then
	# 	wait
	# fi
done

wait
