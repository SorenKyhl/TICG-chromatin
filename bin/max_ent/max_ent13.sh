#! /bin/bash
#SBATCH --job-name=maxent13
#SBATCH --output=logFiles/maxent13.out
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
trust_region=10
resources=~/TICG-chromatin/maxent/resources
chipSeqFolder="/home/erschultz/sequences_to_contact_maps/chip_seq_data"
epiData="${chipSeqFolder}/fold_change_control/processed"
chromHMMData="${chipSeqFolder}/aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt"
results=~/sequences_to_contact_maps/results

source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/eric/sequences_to_contact_maps"
  # dataFolder="/home/eric/dataset_test"
  finalSimProductionSweeps=1000
  numIterations=3
  scratchDir='/home/eric/scratch'
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dir='/project2/depablo/erschultz'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

STARTTIME=$(date +%s)
i=12000
dataset='dataset_09_21_21'
sample=23
for method in 'PCA' 'nmf'
do
  for k in 1
  do
    max_ent
  done
done

for method in 'PCA' 'k_means' 'nmf'
do
  for k in 2 4 6
  do
    max_ent
  done
done


wait

wait

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
