#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup
m=1024
dataset=single_cell_nagano_imputed
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

k=0
nSweeps=100000
dumpFrequency=10000
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
bondLength=28

sample=244
m=1024
# chiDiagMethod="${dataFolder}/samples/sample${sample}/none/k0/replicate1/chis_diag_edit_zero.txt"
# i=244_edit_zero
# run &
#
# chiDiagMethod="${dataFolder}/samples/sample${sample}/none/k0/replicate1/chis_diag_edit2.txt"
# i=244_edit2
# run &
#
# chiDiagMethod="${dataFolder}/samples/sample${sample}/none/k0/replicate1/chis_diag_edit.txt"
# i=244_edit
# run &
