#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=1024
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed='12'
seqSeed='21'
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
nSweeps=1000
dumpFrequency=100
dumpStatsFrequency=100
trackContactMap='false'
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
chiDiagMethod="linear"

# this tests if grid move is a problem, and if the TICG simulations have any systematic bias as a function of time
# i=10
# maxDiagChi=10
# run &
# i=20
# maxDiagChi=20
# run &
# i=2
# gridMoveOn='false'
# run &

i=301
run &

i=302
diagBins=64
run &

i=303
useD='true'
run &

i=304
diagBins=32
useD='false'
useS='true'
run &

i=305
useD='false'
useS='false'
useE='true'
run &

i=306
useS='true'
useD='true'
useE='false'
run &

i=307
useS='false'
useD='true'
useE='true'
run &





wait
