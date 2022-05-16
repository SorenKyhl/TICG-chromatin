#! /bin/bash
k=0
m=500
dataFolder="/home/erschultz/dataset_test3"
scratchDir='/home/erschultz/scratch'
useE='false'
useS='false'
startSample=1
relabel='none'
diag='false'
nSweeps=500000
pSwitch=0.05
maxDiagChi=5
chiSeed='none'
chi='none'
minChi=-2
maxChi=1
fillDiag='none'
overwrite=1
dumpFrequency=10000
TICGSeed='none'
npySeed='none' # for get_seq
method='random'
exclusive='false'
e='none'
s='none'
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


method='random'
nSweeps=500000
diag='true'
maxDiagChi=2
for i in $( seq 1 10 )
do
	echo "i=${i}, maxDiagChi=${maxDiagChi}"
	run &
	maxDiagChi=$(( $maxDiagChi + 2 ))
done

wait
