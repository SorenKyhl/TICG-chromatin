#! /bin/bash
#SBATCH --job-name=maxent12
#SBATCH --output=logFiles/maxent12.out
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
  finalSimProductionSweeps=1000
  numIterations=0
  scratchDir='/home/eric/scratch'
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dir='/project2/depablo/erschultz'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

STARTTIME=$(date +%s)
i=11000
dataset='dataset_09_21_21'
sample=14
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

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
