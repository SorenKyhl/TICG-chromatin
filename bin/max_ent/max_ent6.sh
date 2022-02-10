#! /bin/bash
#SBATCH --job-name=maxent6
#SBATCH --output=logFiles/maxent6.out
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
  numIterations=0
  finalSimProductionSweeps=200000
  equilibSweeps=1000
  productionSweeps=5000
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dir='/project2/depablo/erschultz'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

STARTTIME=$(date +%s)
i=5000
dataset='dataset_11_03_21'
sample=1751

for method in 'random' 'PCA'
do
  for k in 1 2 4 6
  do
    max_ent
  done
done

for method in  'k_means'
do
  for k in 2 4 6
  do
    max_ent
  done
done

method='ground_truth-x'
k=3
max_ent

method='ground_truth-psi'
k=4
max_ent

method='ground_truth'
useE='true'
max_ent

method='GNN'
modelID=42
useE='true'
max_ent

wait


# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
