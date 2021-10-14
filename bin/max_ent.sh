#! /bin/bash
#SBATCH --job-name=TICG_maxent
#SBATCH --output=logFiles/maxent.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=2000

m=1024
samples='40-1230-1718'
dataFolder='/project2/depablo/erschultz/dataset_08_26_21'
productionSweeps=50000
equilibSweeps=10000
goalSpecified=1
numIterations=100 # iteration 1 + numIterations is production run to get contact map
overwrite=1
scratchDir='/scratch/midway2/erschultz/TICG_maxent'
modelType='ContactGNNEnergy'
modelID='23'

source activate python3.8_pytorch1.8.1_cuda10.2
module load jq

STARTTIME=$(date +%s)
i=1
for sample in 40 1230 1718
do
  for k in 1 2
  do
    # 'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'epigenetic'
    for method in 'GNN'
    do
      if [ $method == "GNN" ]
      then
        methodFolder="${method}-${modelID}"
      else
        methodFolder=${method}
      fi
      ofile="${dataFolder}/samples/sample${sample}/${methodFolder}/k${k}"
      scratchDirI="${scratchDir}_${i}"
      mkdir -p $scratchDirI
      cd $scratchDirI
      ~/TICG-chromatin/bin/max_ent_inner.sh $m $k $sample $dataFolder $productionSweeps $equilibSweeps $goalSpecified $numIterations $overwrite $scratchDirI $method $modelType $modelID $ofile > bash.log &
      mv bash.log "${ofile}/bash.log"
      i=$(($i+1))
    done
  done
done

wait

python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small "true"

ENDTIME=$(date +%s)
echo "total time: $(($ENDTIME-$STARTTIME)) seconds"
