#! /bin/bash
#SBATCH --job-name=TICG_maxent
#SBATCH --output=TICG_maxent.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

m=1024
sample=1230
samples='40-1230'
dataFolder='/project2/depablo/erschultz/dataset_08_29_21'
productionSweeps=50000
equilibSweeps=10000
goalSpecified=1
numIterations=100 # iteration 1 + numIterations is production run to get contact map
overwrite=1
scratchDir='/scratch/midway2/erschultz/TICG'
modelType='ContactGNNEnergy'
modelID='28'

source activate python3.8_pytorch1.8.1_cuda10.2
module load jq

# initialize log files
for method in 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'GNN' 'epigenetic' 'ChromHMM'
do
  ofile="/home/erschultz/TICG-chromatin/logFiles/TICG_${method}.log"
  rm $ofile
  touch $ofile
done

STARTTIME=$(date +%s)
for k in 2
do
  #'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'epigenetic'
  for method in 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf'
  do
    ofile="/home/erschultz/TICG-chromatin/logFiles/TICG_${method}.log"
    ~/TICG-chromatin/bin/max_ent_inner.sh $m $k $sample $dataFolder $productionSweeps $equilibSweeps $goalSpecified $numIterations $overwrite "${scratchDir}_${method}" $method $modelType $modelID >> $ofile &
  done
  wait
done

python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

ENDTIME=$(date +%s)
echo "total time: $(($ENDTIME-$STARTTIME)) seconds"

# clean up scratch
for method in 'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'epigenetic'
do
  rm -d "${scratchDir}_${method}"
done
