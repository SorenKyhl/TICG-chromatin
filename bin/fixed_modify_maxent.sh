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

k=10
nSweeps=50000
dumpFrequency=10000
TICGSeed=1464
diag='true'
dense='false'
diagBins=512
nSmallBins=64
smallBinSize=1
bigBinSize=-1
nBigBins=-1
useL='true'
useD='true'
useS='true'

bondLength=140
phiChromatin=0.03
beadVol=130000

jobs=0
waitCount=0
for sample in 221
do
	dataFolder="${baseDataFolder}/samples/sample${sample}/optimize_grid_b_140_phi_0.03-max_ent"
	# initConfig="${dataFolder}/iteration15/equilibrated.txt"
	# gridSize="${sampleFolder}/none/k0/replicate1/grid_size.txt"
	gridSize="${dataFolder}/resources/config.json"

	chiMethod="${dataFolder}/chis.txt"
	chiDiagMethod="${dataFolder}/chis_diag.txt"
	seqMethod="${dataFolder}/resources/x.npy"
	i="${sample}_copy"
	dense='true'
	diagBins=80
	run &

	# chiMethod="${dataFolder}/chis_half.npy"
	# i="${sample}_chi_half_zero"
	# run &

	# chiMethod="${dataFolder}/chis.txt"
	#
	# chiDiagMethod="${dataFolder}/chis_diag_double.npy"
	# i="${sample}_double_diag"
	# run &


	# chiDiagMethod="zeros"
	# i="${sample}_zero"
	# run &

	# chiDiagMethod="${dataFolder}/fitting/chis_diag_edit.txt"
	# i="${sample}_edit"
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
	# chiDiagMethod="${dataFolder}/fitting2/linear_fit.txt"
	# i="${sample}_linear"
	# run &
	# #
	# # chiDiagMethod="${dataFolder}/log_fit.txt"
	# # i=log
	# # run &
	# #
	# chiDiagMethod="${dataFolder}/fitting2/poly2_log_fit.txt"
	# i="${sample}_poly2_log"
	# run &
	# #
	# chiDiagMethod="${dataFolder}/fitting2/poly3_log_fit.txt"
	# i="${sample}_poly3_log"
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting2/poly4_log_fit.txt"
	# i="${sample}_poly4_log"
	# run &

	# chiDiagMethod="${dataFolder}/fitting2/poly4_log_fit.txt"
	# i="${sample}_poly4_log"
	# run &

	chiDiagMethod="${dataFolder}/fitting2/poly6_log_fit.txt"
	i="${sample}_poly6_log"
	run &
	#
	# chiDiagMethod="${dataFolder}/fitting2/poly6_log_fit_edit.txt"
	# i="${sample}_poly6_log_edit"
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting2/poly6_log_fit_edit2.txt"
	# i="${sample}_poly6_log_edit2"
	# run &

	jobs=$(( $jobs + 1 ))
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
