#! /bin/bash
#SBATCH --job-name=maxent9
#SBATCH --output=logFiles/maxent9.out
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
  # dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  finalSimProductionSweeps=1000
  numIterations=0
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=8000
dataset='dataset_04_26_22'
useE='true'
method='GNN'
modelID=150
for sample in 1 2 3 4 5 6 7 8
do
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
