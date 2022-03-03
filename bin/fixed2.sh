#! /bin/bash
k=2
m=500
dataFolder="/home/eric/dataset_test"
scratchDir='/home/eric/scratch'
useE='false'
useS='false'
startSample=1
relabel='none'
diag='false'
nSweeps=100000
pSwitch=0.05
maxDiagChi=0.2
chiSeed='none'
minChi=-1
maxChi=-1
fillDiag='none'
overwrite=1
dumpFrequency=1000
TICGSeed='none'
npySeed='none' # for get_seq
method='random'
exclusive='false'

source activate python3.8_pytorch1.8.1_cuda11.1

source ~/TICG-chromatin/bin/random/random_fns.sh

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

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

chi="polynomial"
pSwitch=0.1
k=6
i=130
nSweeps=100
run &

# pSwitch=0.05
# i=131
# run &
#
# pSwitch=0.01
# i=132
# run &
#
# pSwitch=0.005
# i=133
# run &

wait
