#! /bin/bash
#SBATCH --job-name=maxent14
#SBATCH --output=logFiles/maxent14.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/eric/sequences_to_contact_maps"
  scratchDir='/home/eric/scratch'
  numIterations=4
  finalSimProductionSweeps=1000
  equilibSweeps=1000
  productionSweeps=1000
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=13000
dataset='dataset_09_21_21'
sample=8
gamma=0.001
trust_region=100
mode='both'
diag='true'
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
