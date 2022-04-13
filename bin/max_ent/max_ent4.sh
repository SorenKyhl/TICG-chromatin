#! /bin/bash
#SBATCH --job-name=maxent4
#SBATCH --output=logFiles/maxent4.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh
module unload gcc
module load gcc/10.2.0

if [ $local = 'true' ]
then
  dir="/home/eric/sequences_to_contact_maps"
  scratchDir='/home/eric/scratch'
  # numIterations=1
  # finalSimProductionSweeps=5000
  # equilibSweeps=1000
  # productionSweeps=5000
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  source activate python3.9_pytorch1.9_cuda10.2
fi

STARTTIME=$(date +%s)
i=3000
dataset='dataset_01_17_22'
useE='true'
modelID=109
for method in 'GNN'
do
  for sample in 1
  # 2 3 4 5 6 7 8
  do
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
