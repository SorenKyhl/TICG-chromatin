#! /bin/bash
#SBATCH --job-name=maxent2
#SBATCH --output=logFiles/maxent2.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

m=1024
k='none'
samples='40-1230-1718'
productionSweeps=50000
finalSimProductionSweeps=1000000
equilibSweeps=10000
goalSpecified='true'
numIterations=100 # iteration 1 + numIterations is production run to get contact map
overwrite=1
modelType='ContactGNNEnergy'
local='false'
useE='false'
useS='false'
useGroundTruthChi='false'
useGroundTruthDiagChi='true'
useGroundTruthSeed='false'
mode="plaid"
gamma=0.00001
gammaDiag=0.00001
resources=~/TICG-chromatin/maxent/resources
chipSeqFolder="/home/erschultz/sequences_to_contact_maps/chip_seq_data"
epiData="${chipSeqFolder}/fold_change_control/processed"
chromHMMData="${chipSeqFolder}/aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt"
results=~/sequences_to_contact_maps/results

source ~/TICG-chromatin/bin/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/eric/sequences_to_contact_maps"
  # dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  numIterations=0
  finalSimProductionSweeps=5000
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dir='/project2/depablo/erschultz'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

STARTTIME=$(date +%s)
i=1000
dataset='dataset_01_19_22'
sample=1320

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
k=4
max_ent

method='ground_truth'
useE='true'
max_ent

# method='GNN'
# modelID=71
# useE='true'
# max_ent

wait

python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
