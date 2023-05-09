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

k=0
nSweeps=50000
dumpFrequency=10000
TICGSeed=1464
dense='false'
diagBins=512
nSmallBins=64
smallBinSize=1
bigBinSize=-1
nBigBins=-1
useL='false'
useS='false'

bondLength=140
phiChromatin=0.03
beadVol=130000


jobs=0
waitCount=0
for sample in 221
do
	dataFolder="${baseDataFolder}/samples/sample${sample}/optimize_grid_b_140_phi_0.03-max_ent"
	initConfig="${dataFolder}/iteration15/equilibrated.txt"
	# gridSize="${sampleFolder}/none/k0/replicate1/grid_size.txt"
	gridSize="${dataFolder}/resources/config.json"

	chiMethod="none"
	seqMethod="none"

	chiDiagMethod="${dataFolder}/chis_diag.txt"
	diag='true'
	i="${sample}_k0"
	dense='true'
	useD='true'
	diagBins=80
	# run &


	chiDiagMethod="zeros"
	i="${sample}_k0_zero"
	run &

	# chiDiagMethod="${dataFolder}/fitting2/poly4_log_fit.txt"
	# i="${sample}_k0_poly4_log"
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting2/poly6_log_fit.txt"
	# i="${sample}_k0_poly6_log"
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting2/poly6_log_fit_edit.txt"
	# i="${sample}_k0_poly6_log_edit"
	# run &
	#
	# chiDiagMethod="${dataFolder}/fitting2/poly6_log_fit_edit2.txt"
	# i="${sample}_k0_poly6_log_edit2"
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
