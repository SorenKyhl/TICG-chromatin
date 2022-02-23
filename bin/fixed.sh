#! /bin/bash
k=2
m=800
dataFolder="/home/eric/dataset_test"
scratchDir='/home/eric/scratch'
useE='false'
useS='false'
startSample=1
relabel='none'
diag='false'
nSweeps=10000
pSwitch=0.05
maxDiagChi=0.2
chiSeed='none'
minChi=-1
maxChi=-1
fillDiag='none'
overwrite=1
dumpFrequency=1
TICGSeed='none'
npySeed='12' # for get_seq
method='random'
exclusive='true'

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

i=30
method='block-A100-B600-A100'
diag='true'
for j in -3 -2.5 -2 -1.5 -1 -0.5 0
do
	chi="${j}&0\\0&-1"
	run &
	i=$(($i+1))
done


wait
