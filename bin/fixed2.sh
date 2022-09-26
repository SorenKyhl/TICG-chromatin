#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=1024
dataFolder="/home/erschultz/sequences_to_contact_maps/dataset_04_27_22"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed='31'
seqSeed='31'
chiMethod="${dataFolder}/samples/sample1/chis.npy"
seqMethod="${dataFolder}/samples/sample1/psi.npy"
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

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

k=10
nSweeps=1000000
dumpFrequency=10000
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

# chiDiagMethod="zero"
# i=1_zero
# run &

chiDiagMethod="linear"
i=1_10_long
run &

maxDiagChi=20
i=1_20_long
run &
#
# maxDiagChi=20
# chiDiagMidpoint=30
# chiDiagSlope=100
# chiDiagMethod="logistic"
# i=1_logistic20
# run &
#
# maxDiagChi=30
# i=1_logistic30
# run &
#
maxDiagChi=10
chiDiagMethod="log"
i=1_log10_long
run &

maxDiagChi=20
i=1_log20_long
run &

wait
