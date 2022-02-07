#! /bin/bash
k=2
m=1500
dataFolder="/home/eric/dataset_test"
scratchDir='/home/eric/scratch'
useE='false'
useS='false'
startSample=1
relabel='none'
diag='true'
nSweeps=1000000
pSwitch=0.02
maxDiagChi=0.2
chiSeed='none'
minChi=-1
maxChi=-1
fillDiag='none'
overwrite=1
dumpFrequency=1000 # how often to dump contact map
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
}

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

method='random'
exclusive='true'
i=100
for chi_i in -3 -2 -1 0 1
do
	chi="${chi_i}&0\\0&-1"
	run &
	i=$(($i + 1))
done

method='block-A200-B1100-A200'
exclusive='false'
i=110
for chi_i in -3 -2.5 -2 -1.5 -1
do
	chi="${chi_i}&0\\0&-1"
	run &
	i=$(($i + 1))
done


wait
