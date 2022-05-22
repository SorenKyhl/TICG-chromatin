#! /bin/bash
#SBATCH --job-name=maxent6
#SBATCH --output=logFiles/maxent6.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/eric/sequences_to_contact_maps"
  scratchDir='/home/eric/scratch'
  numIterations=80
  # finalSimProductionSweeps=2000
  # equilibSweeps=1000
  # productionSweeps=5000
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=5000
dataset='dataset_05_18_22'
mode='both'
method='PCA-normalize'
for sample in 1 2 3 4
do
  for k in 2 4 6 8
  do
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
