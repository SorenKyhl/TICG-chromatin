#! /bin/bash
#SBATCH --job-name=maxent9
#SBATCH --output=logFiles/maxent9.out
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
  # dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  finalSimProductionSweeps=1000
  numIterations=0
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=8000
dataset='dataset_09_21_21'
sample=1
gamma=0.001
trust_region=100
mode='both'
diag='false'
for method in 'PCA-normalize' 'nmf'
do
  for k in 1
  do
    max_ent
  done
done

for method in 'PCA-normalize' 'k_means' 'nmf'
do
  for k in 2 4 6
  do
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
