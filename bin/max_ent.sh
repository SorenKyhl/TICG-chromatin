#! /bin/bash
#SBATCH --job-name=TICG_maxent
#SBATCH --output=logFiles/maxent.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=2000

m=1024
samples='40-1230-1718-1201-1202-1203'
dataFolder='/project2/depablo/erschultz/dataset_08_24_21'
productionSweeps=50000
equilibSweeps=10000
goalSpecified=0
numIterations=0 # iteration 1 + numIterations is production run to get contact map
overwrite=1
modelType='ContactGNNEnergy'
modelID='22'
local='true'
binarize='false'
normalize='false'
useEnergy='true'

if [ $local = 'true' ]
then
  dataFolder="/home/eric/sequences_to_contact_maps/dataset_08_24_21"
  scratchDir='/home/eric/scratch'
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2
fi

format_method () {
  if [ $method == "GNN" ]
  then
    methodFolder="${method}-${modelID}"
  else
    methodFolder=${method}
  fi

  if [ $useEnergy = 'true' ]
  then
    methodFolder="${methodFolder}-S"
  fi

  # normalize and binarize are mutually exclusive
  if [ $normalize = 'true' ]
  then
    methodFolder="${methodFolder}-normalize"
  elif [ $binarize = 'true' ]
  then
    methodFolder="${methodFolder}-binarize"
  fi

}

module load jq

STARTTIME=$(date +%s)
i=1
for sample in 40
do
  for k in 2
  do
    # 'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'epigenetic'
    for method in 'ground_truth'
    do
      scratchDirI="${scratchDir}/TICG_maxent${i}"
      mkdir -p $scratchDirI
      cd $scratchDirI

      format_method

      ofile="${dataFolder}/samples/sample${sample}/${methodFolder}/k${k}"
      ~/TICG-chromatin/bin/max_ent_inner.sh $m $k $sample $dataFolder $productionSweeps $equilibSweeps $goalSpecified $numIterations $overwrite $scratchDirI $method $modelType $modelID $ofile $local $binarize $normalize $useEnergy > bash.log &
      mv bash.log "${ofile}/bash.log"
      i=$(($i+1))
    done
  done
done

wait

if [ $local = 'false' ]
then
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small "true"
fi

ENDTIME=$(date +%s)
echo "total time: $(( $ENDTIME - $STARTTIME )) seconds"
