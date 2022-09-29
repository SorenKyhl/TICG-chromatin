#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=1024
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed='31'
seqSeed='31'
chiMethod="/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1/chis.npy"
seqMethod="/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1/psi.npy"
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
dumpStatsFrequency=100
trackContactMap='true'
TICGSeed=10
dense='false'
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
chiDiagMethod="linear"
# i=1
# run &
#
# i=2
# gridMoveOn='false'
# run &
#
# i=3
# gridMoveOn='true'
# updateContactsDistance='true'
# run &
#
# i=4
# gridMoveOn='false'
# updateContactsDistance='true'
# run &


# this tests time comparison of various energy formalisms
gridMoveOn='true'
trackContactMap='false'
chiDiagMethod="linear"
updateContactsDistance='true'

# baseline
i=5
run &

# 2x bins
i=6
diagBins=64
run &

# 4x bins
i=7
diagBins=128
run &

# s matrix
i=8
diagBins=32
useE='false'
useS='true'
run &

# e matrix
i=9
useE='true'
useS='false'
run &

# s+d matrix
i=10
useE='false'
useD='true'
useS='true'
run &

# e+d matrix
i=11
useE='true'
useD='true'
useS='false'
run &

# i=12
# chiDiagMethod='linear'
# useE='true'
# useD='false'
# useS='false'
# run &
# #
# i=13
# constantChi=2
# run &
#
# i=14
# constantChi=0
# chiDiagConstant=2
# run &
#
# i=15
# constantChi=0
# chiDiagConstant=0
# eConstant=0
# sConstant=2
# useE='false'
# useS='true'
# run &

wait
