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
m=512
dataset=dataset_test
baseDataFolder="/home/erschultz/${dataset}"
dataFolder=$baseDataFolder
scratchDir='/home/erschultz/scratch'
overwrite=1

k=0
nSweeps=50000
dumpFrequency=1000
TICGSeed=10
chiSeed=3
seqSeed=4
diag='false'
dense='false'
diagBins=512
nSmallBins=64
smallBinSize=1
bigBinSize=-1
nBigBins=-1
bondLength=28
useD='false'
useL='false'
useE='false'
seqMethod='none'
chiMethod="none"
chiDiagMethod='none'

i=4001
jobs=0
waitCount=0
for bondLength in 18 19 20 21 22 23 24 25 26 27 28
do
	for gridSize in 20 22 24 26 28 30 32 34 36
	do
		echo $i bondLength $bondLength gridSize $gridSize
		run &
		i=$(( $i + 1 ))
		jobs=$(( $jobs + 1 ))
		if [ $jobs -gt 16 ]
		then
			echo 'Waiting'
			waitCount=$(( $waitCount + 1 ))
			wait
			jobs=0
		fi
	done
done

echo $waitCount

wait
