#! /bin/bash
k=2
m=800
dataFolder="/home/erschultz/dataset_test"
scratchDir='/home/erschultz/scratch'
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
dumpFrequency=100
TICGSeed='none'
npySeed='12' # for get_seq
method='random'
exclusive='true'

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

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

i=1
chi='none'
diag='true'
run &

i=2
maxDiagChi=10
run &


wait
