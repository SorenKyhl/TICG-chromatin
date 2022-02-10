#! /bin/bash
#SBATCH --job-name=maxent11
#SBATCH --output=logFiles/maxent11.out
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
i=10000
dataset='dataset_11_14_21'
sample=1718

method='ground_truth-x'
k=12
max_ent

# method='ground_truth-psi'
# k=15
# max_ent
#
method='ground_truth'
useE='true'
max_ent

method='GNN'
modelID=50
useE='true'
max_ent

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
