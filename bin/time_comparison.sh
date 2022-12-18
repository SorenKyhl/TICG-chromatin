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
nSweeps=1000000
dumpFrequency=100000
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

# this tests time comparison of various energy formalisms
# params are different for each sample
gridMoveOn='true'
trackContactMap='false'
chiDiagMethod="linear"
updateContactsDistance='false'

i=201
for m in 512 1024 2048
do
	for useD in 'true' 'false'
	do
		for j in 1 2 3
		do
			if [ $useD = 'true' ]
			then
				useE='true'
			else
				useE='false'
			fi
			echo $i $useD $m $useE
			# run &
			i=$(( $i + 1 ))
		done
	done
done
#
# i=204
# useE='true'
# useS='false'
# m=1024
# dense='false'
# diagBins=32
# chiSeed=21
# seqSeed=12
# for diagBins in 32
# do
# 	for useD in 'true'
# 	do
# 		for j in 1
# 		do
# 			echo $i $useD $m $diagBins
# 			run &
#
# 			i=$(( $i + 1 ))
# 		done
# 	done
# done

wait
