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
m=512
dataset=dataset_02_04_23
baseDataFolder="/home/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'
overwrite=1

k=1
nSweeps=50000
dumpFrequency=10000
TICGSeed=10
diag='true'
dense='true'
diagBins=64
nSmallBins=32
smallBinSize=1
bigBinSize=-1
nBigBins=-1
bondLength=16.5
useL='true'
useD='true'
useE='true'

jobs=0
waitCount=0
for sample in 201
do
	dataFolder="${baseDataFolder}/samples/sample${sample}/none/k${k}/replicate1"
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
	# #
	# chiMethod="${dataFolder}/chis_neg_v2.npy"
	# seqMethod="${dataFolder}/resources/x.npy"
	# i="${sample}_neg2_chi"
	# run &
	#
	# chiMethod="${dataFolder}/chis_shuffle.npy"
	# seqMethod="${dataFolder}/resources/x.npy"
	# i="${sample}_shuffle_chi"
	# run &
	#
	# chiMethod="${dataFolder}/chis_zero.npy"
	# seqMethod="${dataFolder}/resources/x.npy"
	# i="${sample}_zero_chi"
	# run &

	# chiMethod="${dataFolder}/chis_eig.npy"
	# seqMethod="${dataFolder}/resources/x_eig.npy"
	# i="${sample}_eig"
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

	# chiMethod="${dataFolder}/chis.txt"
	# seqMethod="${dataFolder}/resources/x.npy"
	# i="${sample}_copy"
	# run &

	seqMethod="none"
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

	jobs=$(( $jobs + 6 ))
	if [ $jobs -gt 15 ]
	then
		echo 'Waiting'
		waitCount=$(( $waitCount + 1 ))
		wait
		jobs=0
	fi

done

wait

echo $waitCount
