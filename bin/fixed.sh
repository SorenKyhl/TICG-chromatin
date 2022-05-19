#! /bin/bash
k=3
m=1024
dataFolder="/home/erschultz/dataset_test2"
scratchDir='/home/erschultz/scratch'
useE='false'
useS='false'
startSample=1
relabel='none'
diag='false'
nSweeps=500000
pSwitch=0.05
lmbda='none'
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

k=4
method='random'
chi='polynomial'
nSweeps=1000000
dumpFrequency=1000
diag='true'
maxDiagChi=10
useE='true'
m=512
for i in $( seq 1 3 )
do
	echo "i=${i}, m=${m}"
	run &
	if [ $( expr $i % 3 ) -eq 0 ]
	then
		m=$(( $m * 2 ))
	fi
done


wait
