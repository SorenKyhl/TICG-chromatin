#! /bin/bash
#SBATCH --job-name=fds
#SBATCH --output=logFiles/fixed_diag_dataset.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000

source ~/TICG-chromatin/bin/random/random_fns.sh
source activate python3.9_pytorch1.9_cuda10.2

param_setup
k=4
m=1024
dataFolder="/project2/depablo/erschultz/dataset_9_26_22"
scratchDir='/home/erschultz/scratch-midway2'
relabel='none'
lmbda='none'
pSwitch=0.05
chiSeed='none'
seqSeed='none'
chiMethod='random'
minChi=-2
maxChi=2
fillDiag='none'
overwrite=1
dumpFrequency=10000

source activate python3.8_pytorch1.8.1

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

nSweeps=500000
dumpFrequency=5000
TICGSeed=10
chiDiagMethod='logistic'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
nBigBins=-1
bigBinSize=-1
diagCutoff='none'
phiChromatin=0.06
diagStart=0
bondLength=20

i=0
jobs=0
waitCount=0
for chiDiagSlope in 2 4 6 8 10 12 15 20 25 30 35 40
do
	for maxDiagChi in 2 3 4 5 6 7 8 10 12 14
	do
		for chiDiagMidpoint in 20 25 30 35 40 45 50
		do
			for range in 6 4 2
				do
					minChi=$(( -1 * $range ))
					maxChi=$range

		   		i=$(( $i + 1 ))
			  	echo $i 'chiDiagSlope' $chiDiagSlope 'maxDiagChi' $maxDiagChi 'chiDiagMidpoint' $chiDiagMidpoint 'minMaxChi' $minChi $maxChi
			  	# run &


					jobs=$(( $jobs + 1 ))
					if [ $jobs -gt 19 ]
					then
						echo 'Waiting'
						waitCount=$(( $waitCount + 1 ))
						wait
						jobs=0
					fi
				done
		  done
	done
done

echo $waitCount

wait
