#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
k=0
m=1024
dataFolder="/home/erschultz/dataset_test_log2"
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
chiDiagMethod='log'
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
for chiDiagSlope in 10 40 80 100 200 300 400 500 600 700 800 900 1000 1200 1400 1600
do
	for chiDiagConstant in -20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20
	do
		for chiDiagScale in 2 4 6 8 10 12 14 16 18 20 22 24
		do
	   		i=$(( $i + 1 ))
		  	echo $i 'chiDiagSlope' $chiDiagSlope 'constant' $chiDiagConstant 'scale' $chiDiagScale 'bond_length' $bondLength
		  	run &

				jobs=$(( $jobs + 1 ))
				if [ $jobs -gt 18 ]
				then
					echo 'Waiting'
					wait
					jobs=0
				fi
		  done
	done
done

wait
