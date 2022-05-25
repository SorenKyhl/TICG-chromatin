#! /bin/bash
#SBATCH --job-name=maxent11
#SBATCH --output=logFiles/maxent11.out
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
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  numIterations=60
  # finalSimProductionSweeps=1000
  equilibSweeps=5000
  productionSweeps=200000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=10000
dataset='dataset_05_18_22'
useE='true'
method='GNN'
modelID=150
mode='diag'
for sample in 14
# 15 16 17 18 19
do
  max_ent
done

wait


ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
