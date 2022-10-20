#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=1024
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed='none'
seqSeed='none'
chiMethod="random"
seqMethod="random"
minChi=-0.4
maxChi=0.4
fillDiag='none'
overwrite=1

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


k=4
nSweeps=10000
dumpFrequency=1000
dumpStatsFrequency=100
trackContactMap='true'
TICGSeed=10
dense='true'
diagBins=32
maxDiagChi=10
nSmallBins=16
smallBinSize=4
bigBinSize=-1
nBigBins=-1
diagStart=0
diagCutoff='none'
bondLength=28

# this tests if grid move is a problem, and if the TICG simulations have any systematic bias as a function of time
# chiDiagMethod="linear"
# i=10
# maxDiagChi=10
# run &
# i=20
# maxDiagChi=20
# run &
# i=2
# gridMoveOn='false'
# run &

# this tests time comparison of various energy formalisms
gridMoveOn='true'
trackContactMap='false'
chiDiagMethod="linear"
updateContactsDistance='false'

# i=1
# for m in 1024 2048 4096 512
# do
# 	for useD in 'true' 'false'
# 	do
# 		for j in 1 2 3
# 		do
# 			if [ $useD = 'true' ]
# 			then
# 				useE='true'
# 				echo $i $useD $m $useE
# 				run &
# 			else
# 				useE='false'
# 			fi
# 			i=$(( $i + 1 ))
# 		done
# 	done
# done

i=100
useE='false'
m=1024
dense='false'
diagBins=32
chiSeed=21
seqSeed=12
for diagBins in 32 64 128
do
	for useD in 'true' 'false'
	do
		for j in 1
		do
			echo $i $useD $m $diagBins
			# run &

			i=$(( $i + 1 ))
		done
	done
done

wait
