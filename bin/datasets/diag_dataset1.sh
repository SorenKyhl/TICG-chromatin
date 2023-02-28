#! /bin/bash
#SBATCH --job-name=fds
#SBATCH --output=logFiles/fixed_diag_dataset.out
#SBATCH --time=24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9
sourceFile=$1
source $sourceFile


jobs=0
waitCount=0
for i in {1..500}
do
	echo $i
	run &

	jobs=$(( $jobs + 1 ))
	if [ $jobs -gt 23 ]
	then
		echo 'Waiting'
		waitCount=$(( $waitCount + 1 ))
		wait
		jobs=0
	fi
done


echo $waitCount

wait
