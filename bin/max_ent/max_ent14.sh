#! /bin/bash
#SBATCH --job-name=maxent14
#SBATCH --output=logFiles/maxent14.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz/sequences_to_contact_maps"
  scratchDir='/home/erschultz/scratch'
  numIterations=40
  # finalSimProductionSweeps=1000
  # equilibSweeps=1000
  # productionSweeps=1000
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=13000
dataset='dataset_05_18_22'
mode='both'
method='PCA-normalize'
for sample in 16
do
  for k in 8
  do
    mmax_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
