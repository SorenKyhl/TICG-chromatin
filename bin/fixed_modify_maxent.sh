#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=1024
dataset=dataset_07_20_22
dataFolder="/home/erschultz/sequences_to_contact_maps/${dataset}"
scratchDir='/home/erschultz/scratch'
relabel='none'
lmbda=0.8
chiSeed='31'
seqSeed='31'
chiMethod='zero'
minChi=-0.4
maxChi=0.4
fillDiag='none'
overwrite=1

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

k=5
nSweeps=500000
dumpFrequency=100000
TICGSeed=10
diag='true'
dense='true'
diagBins=32
maxDiagChi=10
nSmallBins=16
smallBinSize=4
bigBinSize=-1
nBigBins=-1
diagStart=0
diagCutoff='none'
bondLength=20

sample=104
m=1024

chiMethod="${dataFolder}/samples/sample${sample}/chis.txt"
seqMethod="${dataFolder}/samples/sample${sample}/x.npy"
useD='true'
useE='true'

chiDiagMethod="${dataFolder}/samples/sample${sample}/chis_diag_edit.txt"
i=104_edit
run &
# #
# chiDiagMethod="${dataFolder}/samples/sample${sample}/none/k0/replicate1/chis_diag_edit.txt"
# i=10_edit
# run &
#
# chiDiagMethod="${dataFolder}/samples/sample${sample}_edit/log_max_fit.txt"
# i=10_logmax
# run &
#
# chiDiagMethod="${dataFolder}/samples/sample${sample}_edit/logistic_fit.txt"
# i=10_logistic
# run &
#
# chiDiagMethod="${dataFolder}/samples/sample${sample}_edit/logistic_fit_manual.txt"
# i=10_logistic_manual
# run &
#
# chiDiagMethod="${dataFolder}/samples/sample${sample}_edit/linear_max_fit.txt"
# i=10_linearmax
# run &
#
# chiDiagMethod="${dataFolder}/samples/sample${sample}_edit/log_fit.txt"
# i=10_log
# run &
