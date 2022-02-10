#! /bin/bash
#SBATCH --job-name=maxent9
#SBATCH --output=logFiles/maxent9.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/eric/sequences_to_contact_maps"
  # dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  finalSimProductionSweeps=1000
  numIterations=0
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dir='/project2/depablo/erschultz'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

STARTTIME=$(date +%s)
i=8000
dataset='dataset_10_27_21'
sample=1230

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
k=2
max_ent

method='ground_truth'
useE='true'
max_ent

method='GNN'
modelID=34
useE='true'
max_ent


wait

python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small "true"

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
