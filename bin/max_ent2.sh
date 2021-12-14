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

max_ent() {
  if [ $useS = 'true' ] || [ $useE = 'true' ]
  then
    useGroundTruthChi='false'
    goalSpecified='false'
    numIterations=0
    k=None
  fi
  modelPath="${results}/${modelType}/${modelID}"
  sampleFolder="${dataFolder}/samples/sample${sample}"
  scratchDirI="${scratchDir}/TICG_maxent${i}"
  mkdir -p $scratchDirI

  seed=$RANDOM
  format_method
  for j in 4 5 6
  do
    scratchDirI="${scratchDir}/TICG_maxent${i}"
    mkdir -p $scratchDirI
    cd $scratchDirI
    max_ent_inner $scratchDirI $j $seed > bash.log &
    i=$(( $i + 1 ))
  done

}

STARTTIME=$(date +%s)
i=1000
dataFolder='/project2/depablo/erschultz/dataset_10_27_21'
sample=40
method='ground_truth'

useS='true'
max_ent


useE='true'
useS='false'
max_ent

useE='false'
useGroundTruthChi='true'
numIterations=0
k=2
max_ent




wait

python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small "true"

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
