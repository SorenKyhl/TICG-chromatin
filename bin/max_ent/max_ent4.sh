#! /bin/bash
#SBATCH --job-name=maxent4
#SBATCH --output=logFiles/maxent4.out
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
  numIterations=4
  finalSimProductionSweeps=5000
  equilibSweeps=1000
  productionSweeps=5000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=3000
dataset='dataset_05_18_22'
useE='true'
mode='diag'
method='GNN'
modelID=150
for sample in 1 2 3
do
  max_ent
done

method='PCA-normalize'
mode='both'
useE='false'
modelID='none'
for sample in 1 2 3
do
  for k in 2 4 6 8
  do
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
