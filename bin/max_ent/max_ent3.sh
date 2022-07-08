#! /bin/bash
#SBATCH --job-name=maxent3
#SBATCH --output=logFiles/maxent3.out
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
  numIterations=2
  finalSimProductionSweeps=1000
  equilibSweeps=1000
  productionSweeps=10000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=2000
dataset='dataset_05_18_22'
useE='true'
m=512

method='GNN'
GNNModelID=150
diagChiMethod='linear'
mode='diag'
for sample in 1 2 3
do
  max_ent
done

method='GNN'
GNNModelID=150
diagChiMethod='mlp'
MLPModelID=10
mode='none'
for sample in 1 2 3
do
  max_ent
done

method='ground_truth'
mode='none'
useGroundTruthDiagChi='true'
for sample in 1 2 3
do
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
