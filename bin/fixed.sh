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
dataset=dataset_11_21_22
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

for sample in 410
# 2201
do
	dataFolder="${baseDataFolder}/samples/sample${sample}/PCA-normalize-E/k${k}/replicate1"
	chiDiagMethod="${dataFolder}/chis_diag.txt"
	chiMethod="${dataFolder}/chis.txt"
	seqMethod="${dataFolder}/resources/x.npy"
	i="${sample}_copy"
	run &

done

wait
