#! /bin/bash
#SBATCH --job-name=maxent1
#SBATCH --output=logFiles/maxent1.out
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
  numIterations=3
  finalSimProductionSweeps=1000
  productionSweeps=1000
  equilibSweeps=200
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_05_12_22'
useE='true'
method='ground_truth'
for mode in 'plaid' 'diag'
do
  for sample in 1
   # 2 3 4
  do
    for k in 'none'
    do
      max_ent
    done
  done
done

mode='diag'
method='GNN'
for modelID in 149 150
do
  for sample in 1
  # 2 3 4
  do
    for k in 'none'
    do
      max_ent
    done
  done
done



wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
