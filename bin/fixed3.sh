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
nSweeps=10000
pSwitch=0.05
maxDiagChi=0.1
chiSeed='none'
minChi=-1
maxChi=-1
fillDiag='none'
overwrite=1
dumpFrequency=100
TICGSeed='none'
npySeed='12' # for get_seq
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
	rm $scratchDiri
}

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

chi="polynomial"
for i in 1 2 3 4 5
do
	run &
done

wait
