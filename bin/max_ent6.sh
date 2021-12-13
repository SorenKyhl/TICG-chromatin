#! /bin/bash
#SBATCH --job-name=maxent6
#SBATCH --output=logFiles/maxent6.out
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
binarize='false'
normalize='false'
useE='false'
useS='false'
useGroundTruthChi='false'
useGroundTruthDiagChi='true'
useGroundTruthSeed='false'
correctEnergy='false'
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
  dataFolder="/home/eric/sequences_to_contact_maps/dataset_11_14_21"
  # dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dataFolder='/project2/depablo/erschultz/dataset_11_03_21'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

STARTTIME=$(date +%s)
i=5000
dataFolder='/project2/depablo/erschultz/dataset_12_11_21'
for k in 2 4
do
  for sample in 40 1230 1718
  do
    for method in 'random'
    do
      # 'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'epigenetic' 'kPCA-x' 'kPCA-y'
      max_ent
    done
  done
done

wait

python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small "true"

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
