#! /bin/bash
k=3
m=1024
dataFolder="/home/erschultz/sequences_to_contact_maps/dataset_04_27_22"
scratchDir='/home/erschultz/scratch'
useE='false'
useS='false'
startSample=1
relabel='none'
diag='false'
nSweeps=500000
pSwitch=0.05
maxDiagChi=5
chiSeed=89
chi='none'
minChi=-2
maxChi=1
fillDiag='none'
overwrite=1
dumpFrequency=10000
TICGSeed=14
npySeed=12 # for get_seq
method='random'
exclusive='false'
e='none'
chiConstant=0
chiDiagConstant=0
sConstant=0

source activate python3.8_pytorch1.8.1

source ~/TICG-chromatin/bin/random/random_fns.sh

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

# cd ~/TICG-chromatin/srcs
# make
# mv TICG-engine ..

k=1
nSweeps=1000000
diag='true'
maxDiagChi=10
useE='true'
s="${dataFolder}/samples/sample1/s_pca_3_min_MSE.npy"
i=0
run &

# sConstant=10
# chiDiagConstant=-454.613
# i=13
# run &
#
# i=14
# sConstant=-10
# chiDiagConstant=454.613
# run &

# i=15
# sConstant=5
# chiDiagConstant=-5
# run &

wait
