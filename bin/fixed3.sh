#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh
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

param_setup
m=1024
dataFolder="/home/erschultz/dataset_test_vbead"
scratchDir='/home/erschultz/scratch'
relabel='none'
chiSeed='12'
seqSeed='21'
chiMethod="none"
seqMethod="none"
chiDiagMethod='none'
fillDiag='none'
overwrite=1

k=0
nSweeps=100000
dumpFrequency=10000
dumpStatsFrequency=100
trackContactMap='false'
TICGSeed=10
bondLength=453.8
gridSize=431.1

# bondLength=36.895
# gridSize=35.05
# beadVol=2600
# m=512
# i=3
# run &
#
# bondLength=16.5
# gridSize=15.675
# beadVol=520
# m=2560
# i=4
# run &

# bondLength=442
# gridSize=453.7
# beadVol=130000
# m=1024
# i=6
# run &


# i=101
# m=1024
# jobs=0
# for gridSize in 28.7 22.8 23.75 24.7 25.65 26.6 27.55 28.5 29.45 30.4
# do
# 	for beadVol in 200 400 600 800 1200 1600 2000 2400 2800 3200
# 	do
# 		for bondLength in 24 25 26 27 28 29 30 31 32
# 		do
# 			echo $i $beadVol $bondLength
# 			# run &
# 			i=$(( $i + 1 ))
#
# 			jobs=$(( $jobs + 1 ))
# 			if [ $jobs -gt 14 ]
# 			then
# 				echo "Waiting on ${jobs} jobs"
# 				waitCount=$(( $waitCount + 1 ))
# 				wait
# 				jobs=0
# 			fi
# 		done
# 	done
# done

i=201
m=1024
jobs=0
for gridSize in 27 27.5 28 28.5 30 30.5 31
do
	for beadVol in 100 200 300 400 500 600
	do
		for bondLength in 28 28.5 29 29.5 30 31.5
		do
			echo $i $beadVol $bondLength
			run &
			i=$(( $i + 1 ))

			jobs=$(( $jobs + 1 ))
			if [ $jobs -gt 18 ]
			then
				echo "Waiting on ${jobs} jobs"
				waitCount=$(( $waitCount + 1 ))
				wait
				jobs=0
			fi
		done
	done
done

# m=1024
# beadVol=520
#
#
# bondLength=24
# gridSize=22.8
# i=21
# run &
#
# bondLength=26
# gridSize=24.7
# i=22
# run &
#
# bondLength=28
# gridSize=26.6
# i=23
# run &
#
# bondLength=30
# gridSize=28.5
# i=24
# run &
#
# bondLength=32
# gridSize=30.4
# i=25
# run &
#
# bondLength=34
# gridSize=32.3
# i=26
# run &

echo $waitCount

wait
