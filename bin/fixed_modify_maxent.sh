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
nSweeps=500000
dumpFrequency=10000
TICGSeed=10
diag='true'
dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
bigBinSize=-1
nBigBins=-1
bondLength=28
useD='true'
useE='true'
m=1024

for sample in 2201
# 2201
do
	dataFolder="${baseDataFolder}/samples/sample${sample}/PCA-normalize-E/k${k}/replicate1"
	chiDiagMethod="${dataFolder}/chis_diag.txt"


	# chiMethod="${dataFolder}/chis.txt"
	# seqMethod="${dataFolder}/resources/x_shuffle.npy"
	# i="${sample}_shuffle_seq"
	# run &
	#
	# chiMethod="${dataFolder}/chis.txt"
	# seqMethod="${baseDataFolder}/samples/sample2202/PCA-normalize-E/k${k}/replicate1/resources/x.npy"
	# i="${sample}_other_pcs"
	# run &
	#
	# chiMethod="${dataFolder}/chis_neg.npy"
	# seqMethod="${dataFolder}/resources/x.npy"
	# i="${sample}_neg_chi"
	# run &
	#
	# chiMethod="${dataFolder}/chis_shuffle.npy"
	# seqMethod="${dataFolder}/resources/x.npy"
	# i="${sample}_shuffle_chi"
	# run &
	#
	# chiMethod="/home/erschultz/dataset_11_21_22/samples/sample1462/chis.npy"
	# seqMethod="${dataFolder}/resources/x.npy"
	# i="${sample}_rand_chi"
	# run &
	#
	# chiMethod="${dataFolder}/chis.txt"
	# seqMethod="/home/erschultz/dataset_11_21_22/samples/sample1462/x.npy"
	# i="${sample}_rand_seq"
	# run &

	chiMethod="${dataFolder}/chis.txt"
	seqMethod="${dataFolder}/resources/x.npy"
	i="${sample}_copy"
	run &

	# chiDiagMethod="${dataFolder}/fitting/chis_diag_edit.txt"
	# i="${sample}_edit"
	# run &

	# chiDiagMethod="${dataFolder}/chis_diag_edit_zero.txt"
	# i="${sample}_edit_zero"
	# run &

	# chiDiagMethod="${dataFolder}/log_max_fit.txt"
	# i=logmax
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting/logistic_fit.txt"
	# i="${sample}_logistic"
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting/logistic_fit_manual.txt"
	# i="${sample}_logistic_manual"
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting/linear_fit.txt"
	# i="${sample}_linear_rand_seq"
	# run &
	# #
	# # chiDiagMethod="${dataFolder}/log_fit.txt"
	# # i=log
	# # run &
	# #
	# chiDiagMethod="${dataFolder}/fitting/poly2_fit.txt"
	# i="${sample}_poly2"
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting/poly3_fit.txt"
	# i="${sample}_poly3"
	# run &
done

wait
