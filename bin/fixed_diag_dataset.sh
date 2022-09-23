#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
k=0
m=1024
dataFolder="/home/erschultz/dataset_test_logistic"
scratchDir='/home/erschultz/scratch'
startSample=1
relabel='none'
diag='false'
lmbda=0.8
maxDiagChi=5
chiSeed='31'
seqSeed='31'
chiMethod='zero'
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

nSweeps=10000
dumpFrequency=5000
TICGSeed=10
chiDiagMethod='logistic'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
nBigBins=-1
bigBinSize=-1
diagCutoff='none'
phiChromatin=0.06
diagStart=0
bondLength=28

i=0
jobs=0
waitCount=0
for chiDiagSlope in 20 30 40 50 60 70 80 90 100 110 120 130 140 150
do
	for maxDiagChi in 23 26 29 32 35 38 41 44 47 50 53 56 59
	do
		for chiDiagMidpoint in 16 18 20 22 24 26 28 30 32 34 36 38
		do
	   		i=$(( $i + 1 ))
		  	echo $i 'chiDiagSlope' $chiDiagSlope 'maxDiagChi' $maxDiagChi 'chiDiagMidpoint' $chiDiagMidpoint
		  	# run &

				jobs=$(( $jobs + 1 ))
				if [ $jobs -gt 18 ]
				then
					echo 'Waiting'
					waitCount=$(( $waitCount + 1 ))
					wait
					jobs=0
				fi
		  done
	done
done

echo $waitCount

wait
