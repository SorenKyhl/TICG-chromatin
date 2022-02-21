#! /bin/bash

# directories
resources=~/TICG-chromatin/maxent/resources
results=~/sequences_to_contact_maps/results

# sweep params
productionSweeps=50000
finalSimProductionSweeps=1000000
equilibSweeps=10000
numIterations=100 # iteration 1 + numIterations is production run to get contact map

# energy params
useE='false'
useS='false'

# general params
overwrite=1
loadChi='false'
project='false'
goalSpecified='true'
modelType='ContactGNNEnergy'
m=-1
diag='false'
diagBins=20

# ground truth params
useGroundTruthChi='false'
useGroundTruthDiagChi='true'
useGroundTruthSeed='false'

# newton's method params
mode="plaid"
gamma=1
trust_region=10

# experimental data
chipSeqFolder="/home/erschultz/sequences_to_contact_maps/chip_seq_data"
epiData="${chipSeqFolder}/fold_change_control/processed"
chromHMMData="${chipSeqFolder}/aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt"

max_ent() {
  if [ $mode = 'plaid' ] || [ $mode = 'both' ]
  then
    if [ $useS = 'true' ] || [ $useE = 'true' ]
    then
      useGroundTruthChi='false'
      goalSpecified='false'
      numIterations=0
      if ! [ $loadChi = 'true' ]
      then
        k='none'
      fi
    fi
    if [ $useGroundTruthChi == 'true' ]
    then
      numIterations=0
      goalSpecified='false'
    fi
  fi

  if [ $mode = 'diag' ] || [ $mode = 'both' ]
  then
    diag='true'
  fi

  dataFolder="${dir}/${dataset}"
  modelPath="${results}/${modelType}/${modelID}"
  sampleFolder="${dataFolder}/samples/sample${sample}"
  scratchDirI="${scratchDir}/TICG_maxent${i}"
  mkdir -p $scratchDirI

  seed=$RANDOM
  format_method
  for j in 1
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
  ofile="${sampleFolder}/${method_fmt}/k${k}/replicate${2}"
  echo $ofile

  # move to scratch
  scratchDirResources="${1}/resources"
  mkdir -p $scratchDirResources

  cd $resources
  init_config="input${m}.xyz"
  if [ -f $init_config ]
  then
    cp $init_config $scratchDirResources
  else
    init_config='none'
  fi

  cd $scratchDirResources
  # generate sequences
  echo "starting get_seq"
  python3 ~/TICG-chromatin/scripts/get_seq.py --method $method_fmt --m $m --k $k --sample $sample --data_folder $dataFolder --plot --save_npy --epigenetic_data_folder $epiData --ChromHMM_data_file $chromHMMData --model_path $modelPath --seed $3 --local $local > seq.log

  # get config
  echo "starting get_config"
  python3 ~/TICG-chromatin/scripts/get_config.py --k $k --m $m --min_chi=-1 --max_chi=1 --save_chi_for_max_ent --goal_specified $goalSpecified --default_config "${resources}/default_config.json" --use_ematrix $useE --use_smatrix $useS --use_ground_truth_chi $useGroundTruthChi --use_ground_truth_diag_chi $useGroundTruthDiagChi --use_ground_truth_TICG_seed $useGroundTruthSeed --TICG_seed $RANDOM --sample_folder $sampleFolder --load_configuration_filename $init_config --diag $diag --diag_bins $diagBins > config.log


  # generate goals
  if [ $goalSpecified = 'true' ]
  then
    echo "starting goal_specified"
    python3 ~/TICG-chromatin/maxent/bin/get_goal_experimental.py --m $m --k $k --contact_map "${sampleFolder}/y.npy" --mode $mode --diag_bins $diagBins > goal.log
  fi

  echo $method_fmt
  # apply max ent with newton's method
  ~/TICG-chromatin/maxent/bin/run.sh $ofile $gamma $trust_region $mode $productionSweeps $equilibSweeps $goalSpecified $numIterations $overwrite $1 $finalSimProductionSweeps

  # run.sh moves all data to $ofile upon completion
  cd $ofile

  # compare results
  prodIt=$(($numIterations+1))
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --k $k --final_it $prodIt --replicate_folder $ofile --save_npy
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${ofile}/y.npy" --y_diag "$sampleFolder/y_diag.npy" --yhat_diag "${ofile}/y_diag.npy"

  echo "\n\n"
}

format_method () {
  method_fmt=$method

  if [ $method == 'GNN' ]
  then
    method_fmt="${method_fmt}-${modelID}"
  fi

  if [ $loadChi = 'true' ]
  then
    method_fmt="${method_fmt}-load_chi"
  fi

  if [ $project = 'true' ]
  then
    method_fmt="${method_fmt}-project"
  fi

  if [ $useGroundTruthChi = 'true' ]
  then
    method_fmt="${method_fmt}-chi"
  fi

  if [ $useS = 'true' ]
  then
    method_fmt="${method_fmt}-S"
  fi

  if [ $useE = 'true' ]
  then
    method_fmt="${method_fmt}-E"
  fi

  # seed
  if [ $useGroundTruthSeed = 'true' ]
  then
  method_fmt="${method_fmt}-seed"
  fi

  # diag
  if [ $diag = 'true' ]
  then
  method_fmt="${method_fmt}-diagOn"
  fi

  echo $method_fmt
}
