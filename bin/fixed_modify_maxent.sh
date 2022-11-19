#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh
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

param_setup
m=1024
dataset=dataset_11_14_22
baseDataFolder="/home/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'
overwrite=1

k=8
nSweeps=100000
dumpFrequency=10000
TICGSeed=10
diag='true'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
bigBinSize=-1
nBigBins=-1
bondLength=28
useD='true'
useE='true'
m=1024

for sample in 6
do
	dataFolder="${baseDataFolder}/samples/sample${sample}/PCA_split-binarizeMean-E/k${k}/replicate1"
	chiMethod="${dataFolder}/chis.txt"
	seqMethod="${dataFolder}/resources/x.npy"

	chiDiagMethod="${dataFolder}/fitting/chis_diag_edit.txt"
	i="${sample}_edit"
	run &

	# chiDiagMethod="${dataFolder}/chis_diag_edit_zero.txt"
	# i="${sample}_edit_zero"
	# run &

	# chiDiagMethod="${dataFolder}/log_max_fit.txt"
	# i=logmax
	# run &
	#
	# chiDiagMethod="${dataFolder}/logistic_fit.txt"
	# i=logistic
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting/logistic_fit_manual.txt"
	# i="${sample}_logistic_manual"
	# run &

	chiDiagMethod="${dataFolder}/fitting/linear_fit.txt"
	i="${sample}_linear"
	run &
	# #
	# # chiDiagMethod="${dataFolder}/log_fit.txt"
	# # i=log
	# # run &
	# #
	chiDiagMethod="${dataFolder}/fitting/poly2_fit.txt"
	i="${sample}_poly2"
	run &
	#
	chiDiagMethod="${dataFolder}/fitting/poly3_fit.txt"
	i="${sample}_poly3"
	run &
done

wait
