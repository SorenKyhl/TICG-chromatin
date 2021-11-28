#! /bin/bash
#SBATCH --job-name=maxent
#SBATCH --output=logFiles/maxent.out
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
useEnergy='false'
useGroundTruthChi='false'
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


if [ $local = 'true' ]
then
  dataFolder="/home/eric/sequences_to_contact_maps/dataset_11_14_21"
  dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dataFolder='/project2/depablo/erschultz/dataset_08_29_21'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

max_ent() {
  modelPath="${results}/${modelType}/${modelID}"
  sampleFolder="${dataFolder}/samples/sample${sample}"
  scratchDirI="${scratchDir}/TICG_maxent${i}"
  mkdir -p $scratchDirI

  seed=$RANDOM
  format_method
  for j in 1 2 3
  do
    scratchDirI="${scratchDir}/TICG_maxent${i}"
    mkdir -p $scratchDirI
    cd $scratchDirI
    max_ent_inner $scratchDirI $j $seed > bash.log &
    i=$(( $i + 1 ))
  done

}

max_ent_inner () {
  # args:
  # 1 = scratchDir
  # 2 = replicate index
  # 3 = seed for get_seq
  ofile="${sampleFolder}/${methodFolder}/k${k}/replicate${2}"

  # move to scratch
  scratchDirResources="${1}/resources"
  mkdir -p $scratchDirResources
  cd $scratchDirResources
  cp "${resources}/input1024.xyz" .

  # get config
  python3 ~/TICG-chromatin/scripts/get_config.py --k $k --m $m --min_chi=-1 --max_chi=1 --save_chi_for_max_ent --goal_specified $goalSpecified --default_config "${resources}/default_config.json" --use_energy $useEnergy --use_ground_truth_chi $useGroundTruthChi --use_ground_truth_seed $useGroundTruthSeed --seed $RANDOM --sample_folder $sampleFolder

  # generate sequences
  python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --k $k --sample $sample --data_folder $dataFolder --plot --save_npy --epigenetic_data_folder $epiData --ChromHMM_data_file $chromHMMData --model_path $modelPath --use_energy $useEnergy --seed $3 --correct_energy $correctEnergy

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
    methodFolder="${methodFolder}-fixed-chi"
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
# for k in 2 4
# do
#   for sample in 40
#   do
#     for method in 'random' 'k_means' 'PCA' 'PCA_split' 'nmf'
#     do
#       # 'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf' 'epigenetic'
#       max_ent
#     done
#   done
# done
# #
# k=4
# method='ground_truth'
# for sample in 40 1230 1718
# do
#   max_ent
# done
#
# dataFolder='/project2/depablo/erschultz/dataset_08_29_21'
# useGroundTruthChi='true'
# method='ground_truth'
# numIterations=0
# goalSpecified='false'
# k=2
# for sample in 40
# do
#   max_ent
# done
#
# useGroundTruthChi='false'
# method='ground_truth'
# numIterations=0
# goalSpecified='false'
# useEnergy='true'
# k='none'
# for sample in 40 1230 1718
# do
#   max_ent
# done
#
# dataFolder='/project2/depablo/erschultz/dataset_08_26_21'
# k='none'
# useEnergy='true'
# goalSpecified='false'
# useGroundTruthChi='false'
# numIterations=0
# correctEnergy='true'
# modelID=58
# method='GNN'
# for sample in 40 1230 1718
# do
#     max_ent
# done

dataFolder='/project2/depablo/erschultz/dataset_08_29_21'
k='none'
useEnergy='true'
goalSpecified='false'
useGroundTruthChi='false'
numIterations=0
correctEnergy='false'
modelID=55
method='GNN'
for sample in 40 1230 1718
do
    max_ent
done
#
# k='none'
# useEnergy='true'
# goalSpecified='false'
# useGroundTruthChi='false'
# numIterations=0
# correctEnergy='false'
# for sample in 40 1230 1718
# do
#   for method in  'ground_truth'
#   # 'ground_truth' 'GNN'
#   do
#     max_ent
#   done
# done

wait

python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small "true"

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
