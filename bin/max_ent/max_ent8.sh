#! /bin/bash
#SBATCH --job-name=maxent8
#SBATCH --output=logFiles/maxent8.out
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
  numIterations=2
  finalSimProductionSweeps=5000
  equilibSweeps=1000
  productionSweeps=5000
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=7000
dataset='dataset_05_18_22'
useE='false'
method='PCA-normalize'
diagChiMethod='mlp'
MLPModelID=10
mode='both'
m=512

for sample in 1 2 3
do
  for k in 2 4 6
  do
    max_ent
  done
done

wait


ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
