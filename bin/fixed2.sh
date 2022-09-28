#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=2048
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed='31'
seqSeed='31'
chiMethod="/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1/chis.npy"
seqMethod="random"
# "/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1/psi.npy"
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
nSweeps=500000
dumpFrequency=1000
dumpStatsFrequency=200
trackContactmap='true'
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
chiDiagMethod="linear"
# i=1
# run &
#
# i=2
# gridMoveOn='false'
# run &

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
trackContactmap='false'
chiDiagMethod="linear"

# # baseline
# i=5
# run &
#
# # 2x bins
# i=6
# diagBins=64
# run &

# 4x bins
i=7
diagBins=128
run &

# # e matrix
# i=8
# diagBins=32
# useE='true'
# run &

wait
