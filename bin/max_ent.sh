#! /bin/bash
#SBATCH --job-name=maxent
#SBATCH --output=logFiles/maxent.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=2000

m=1024
k='none'
samples='40-1230-1718-1201-1202-1203'
dataFolder='/project2/depablo/erschultz/dataset_10_27_21'
productionSweeps=50000
finalSimProductionSweeps=1000000
equilibSweeps=10000
goalSpecified='true'
numIterations=100 # iteration 1 + numIterations is production run to get contact map
overwrite=1
modelType='ContactGNNEnergy'
modelID='34'
local='true'
binarize='false'
normalize='false'
useEnergy='false'
useGroundTruthChi='false'
useGroundTruthSeed='false'
mode="plaid"
gamma=0.00001
gammaDiag=0.00001
resources=~/TICG-chromatin/maxent/resources
chipSeqFolder="/home/erschultz/sequences_to_contact_maps/chip_seq_data"
epiData="${chipSeqFolder}/fold_change_control/processed"
chromHMMData="${chipSeqFolder}/aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt"
results=~/sequences_to_contact_maps/results
modelPath="${results}/${modelType}/${modelID}"


if [ $local = 'true' ]
then
  dataFolder="/home/eric/sequences_to_contact_maps/dataset_10_27_21"
  # dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2
fi


max_ent () {
  sampleFolder="${dataFolder}/samples/sample${sample}"
  if [ $k = 'none' ]
  then
    ofile="${sampleFolder}/${methodFolder}"
  else
    ofile="${sampleFolder}/${methodFolder}/k${k}"
  fi

  # move to scratch
  scratchDirResources="${1}/resources"
  mkdir -p $scratchDirResources
  cd $scratchDirResources
  cp "${resources}/input1024.xyz" .

  # get config
  python3 ~/TICG-chromatin/scripts/get_config.py --k $k --m $m --min_chi=-1 --max_chi=1 --save_chi_for_max_ent --goal_specified $goalSpecified --default_config "${resources}/default_config.json" --use_energy $useEnergy --use_ground_truth_chi $useGroundTruthChi --use_ground_truth_seed $useGroundTruthSeed --seed $RANDOM --sample_folder $sampleFolder

  # generate sequences
  python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --k $k --sample $sample --data_folder $dataFolder --plot --save_npy --epigenetic_data_folder $epiData --ChromHMM_data_file $chromHMMData --model_path $modelPath --use_energy $useEnergy

  # generate goals
  if [ $goalSpecified = 'true' ]
  then
    python3 ~/TICG-chromatin/maxent/bin/get_goal_experimental.py --m $m --k $k --contact_map "${sampleFolder}/y.npy"
  fi

  echo $method
  # apply max ent with newton's method
  ~/TICG-chromatin/maxent/bin/run.sh $ofile $gamma $gammaDiag $mode $productionSweeps $equilibSweeps $goalSpecified $numIterations $overwrite $1 $finalSimProductionSweeps

  # run.sh moves all data to $ofile upon completion
  cd $ofile

  # compare results
  prodIt=$(($numIterations+1))
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${ofile}/iteration${prodIt}/y.npy" --y_diag_instance "$sampleFolder/y_diag_instance.npy" --yhat_diag_instance "${ofile}/iteration${prodIt}/y_diag_instance.npy"

  echo "\n\n"
}

format_method () {
  if [ $method == "GNN" ]
  then
    methodFolder="${method}-${modelID}"
  else
    methodFolder=${method}
  fi

  if [ $useGroundTruthChi = 'true' ]
  then
    methodFolder="${methodFolder}-gt_chi"
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

  # seed
  if [ $useGroundTruthSeed = 'true' ]
  then
  methodFolder="${methodFolder}-seed"
  fi

  echo $methodFolder
}

STARTTIME=$(date +%s)
i=1
# k=4
# for sample in 40
# do
#   for method in 'random' 'k_means' 'PCA' 'PCA_split' 'nmf'
#   do
#     # 'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'epigenetic'
#     scratchDirI="${scratchDir}/TICG_maxent${i}"
#     mkdir -p $scratchDirI
#     cd $scratchDirI
#
#     format_method
#     max_ent $scratchDirI > bash.log &
#     i=$(($i+1))
#   done
# done

k='none'
numIterations=0
useEnergy='true'
goalSpecified='false'
for sample in 40
do
  for method in 'GNN'
  do
    scratchDirI="${scratchDir}/TICG_maxent${i}"
    mkdir -p $scratchDirI
    cd $scratchDirI

    format_method
    max_ent $scratchDirI > bash.log &
    i=$(($i+1))
  done
done
#
# useEnergy='false'
# useGroundTruthChi='true'
# k=2
# for sample in 40
# do
#   for method in 'ground_truth'
#   do
#     # 'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'epigenetic'
#     scratchDirI="${scratchDir}/TICG_maxent${i}"
#     mkdir -p $scratchDirI
#     cd $scratchDirI
#
#     format_method
#     max_ent $scratchDirI > bash.log &
#     i=$(($i+1))
#   done
# done

wait

if [ $local = 'false' ]
then
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small "true"
fi

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
