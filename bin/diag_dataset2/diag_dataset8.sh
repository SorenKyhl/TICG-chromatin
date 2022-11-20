#! /bin/bash
#SBATCH --job-name=fds8
#SBATCH --output=logFiles/fixed_diag_dataset8.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9
source ~/TICG-chromatin/bin/diag_dataset2/diag_dataset_fns.sh


jobs=0
waitCount=0
for i in {2101..2400}
do
	echo $i
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


echo $waitCount

wait
