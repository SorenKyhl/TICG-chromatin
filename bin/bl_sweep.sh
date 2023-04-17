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
dataset=dataset_bond
baseDataFolder="/home/erschultz/${dataset}"
dataFolder=$baseDataFolder
scratchDir='/home/erschultz/scratch'
overwrite=1

k=0
nSweeps=50000
dumpFrequency=1000
dumpStatsFrequency=100
TICGSeed=10
chiSeed=3
seqSeed=4
diag='false'
dense='false'
gridSize='scale_1.4'
beadVol=260000
useD='false'
useL='false'
useE='false'
seqMethod='none'
chiMethod="none"
chiDiagMethod='none'

i=7
jobs=0
waitCount=0
for bondLength in 131 261 488
do
	echo $i bondLength $bondLength
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

echo $waitCount

wait
