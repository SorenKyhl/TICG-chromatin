#! /bin/bash
#SBATCH --job-name=maxent5
#SBATCH --output=logFiles/maxent5.out
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
  numIterations=1
  finalSimProductionSweeps=5000
  equilibSweeps=1000
  productionSweeps=1000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=4000
dataset='dataset_05_18_22'
useE='true'
method='GNN'
modelID=150
mode='diag'
for sample in 11
do
  max_ent
done


wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
