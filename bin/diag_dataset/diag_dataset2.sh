#! /bin/bash
#SBATCH --job-name=fds2
#SBATCH --output=logFiles/fixed_diag_dataset2.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9
source ~/TICG-chromatin/bin/diag_dataset/diag_dataset_fns.sh

i=630
jobs=0
waitCount=0
for chiDiagSlope in 8 10 12
do
	for maxDiagChi in 2 3 4 5 6 7 8 10 12 14
	do
		for chiDiagMidpoint in 20 25 30 35 40 45 50
		do
			for range in 2 4 6
				do
					minChi=$(( -1 * $range ))
					maxChi=$range

		   		i=$(( $i + 1 ))
			  	echo $i 'chiDiagSlope' $chiDiagSlope 'maxDiagChi' $maxDiagChi 'chiDiagMidpoint' $chiDiagMidpoint 'minMaxChi' $minChi $maxChi
			  	run &

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
