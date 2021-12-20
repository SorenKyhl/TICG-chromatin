#! /bin/bash
max_ent() {
  if [ $useS = 'true' ] || [ $useE = 'true' ]
  then
    useGroundTruthChi='false'
    goalSpecified='false'
    numIterations=0
    k='none'
  fi
  if [ $useGroundTruthChi == 'true' ]
  then
    numIterations=0
    goalSpecified='false'
  fi
  dataFolder="${dir}/${dataset}"
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
  ofile="${sampleFolder}/${method}/k${k}/replicate${2}"

  # move to scratch
  scratchDirResources="${1}/resources"
  mkdir -p $scratchDirResources
  cd $scratchDirResources
  cp "${resources}/input1024.xyz" .

  # get config
  python3 ~/TICG-chromatin/scripts/get_config.py --k $k --m $m --min_chi=-1 --max_chi=1 --save_chi_for_max_ent --goal_specified $goalSpecified --default_config "${resources}/default_config.json" --use_ematrix $useE --use_smatrix $useS --use_ground_truth_chi $useGroundTruthChi --use_ground_truth_diag_chi $useGroundTruthDiagChi --use_ground_truth_TICG_seed $useGroundTruthSeed --TICG_seed $RANDOM --sample_folder $sampleFolder

  # generate sequences
  python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --k $k --sample $sample --data_folder $dataFolder --plot --save_npy --epigenetic_data_folder $epiData --ChromHMM_data_file $chromHMMData --model_path $modelPath --seed $3

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
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${ofile}/iteration${prodIt}/y.npy" --y_diag "$sampleFolder/y_diag.npy" --yhat_diag "${ofile}/iteration${prodIt}/y_diag.npy"

  echo "\n\n"
}

format_method () {
  echo $method

  if [ $useGroundTruthChi = 'true' ]
  then
    method="${method}-chi"
  fi

  if [ $useS = 'true' ]
  then
    method="${method}-S"
    echo here2
  fi

  if [ $useE = 'true' ]
  then
    method="${method}-E"
  fi

  # seed
  if [ $useGroundTruthSeed = 'true' ]
  then
  method="${method}-seed"
  fi

  echo $method
}
