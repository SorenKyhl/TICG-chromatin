#! /bin/bash
#SBATCH --job-name=maxent13
#SBATCH --output=logFiles/maxent13.out
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
  numIterations=1
  finalSimProductionSweeps=5000
  equilibSweeps=1000
  productionSweeps=5000
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=12000
dataset='dataset_09_21_21'
sample=8
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


# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
