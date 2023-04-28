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
m=1024
dataset=dataset_angle
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
beadVol=260000
bondLength=178
gridSize=222
useD='false'
useL='false'
useE='false'
seqMethod='none'
chiMethod="none"
chiDiagMethod='none'
phiChromatin=0.06

i=100
jobs=0
waitCount=0
for kAngle in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0
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
