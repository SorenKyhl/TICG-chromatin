#! /bin/bash
source activate python3.9_pytorch1.9_cuda10.2
module unload gcc # not sure if this is necessary
module load gcc/10.2.0

# directories
resources=~/TICG-chromatin/utils
results=~/sequences_to_contact_maps/results
dir='/project2/depablo/erschultz'
scratchDir='/scratch/midway2/erschultz'

# sweep params
productionSweeps=300000
finalSimProductionSweeps=1000000
equilibSweeps=50000
numIterations=40 # iteration 1 + numIterations is production run to get contact map

# energy params
useE='false'
useS='false'

# general params
overwrite=1
loadChi='false'
project='false'
goalSpecified='true'
GNNModelType='ContactGNNEnergy'
m=-1
k=-1
seqSeed='none'
chiSeed='none'
TICGSeed='none'
diagChiMethod='linear'
diagBins=20
diagPseudobeadsOn='true'

# ground truth params
useGroundTruthChi='false'
useGroundTruthDiagChi='false'
useGroundTruthSeed='false'

# newton's method params
mode="plaid"
trust_region=50
gamma=1
minDiagChi='none'

# experimental data
chipSeqFolder="/home/erschultz/sequences_to_contact_maps/chip_seq_data"
epiData="${chipSeqFolder}/fold_change_control/processed"
chromHMMData="${chipSeqFolder}/aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt"

max_ent_resume(){
  # args:
  # 1 = start iteration
  param_setup
  format_method
  for j in 1
  do
    scratchDirI="${scratchDir}/TICG_maxent${i}"
    mkdir -p $scratchDirI
    cd $scratchDirI
    max_ent_resume_inner $scratchDirI $j $1 >> bash.log &
    i=$(( $i + 1 ))
  done
}

max_ent_resume_inner(){
  # args:
  # 1 = scratchDir
  # 2 = replicate index
  # 3 = start iteration
  ofile="${sampleFolder}/${method_fmt}/k${k}/replicate${2}"
  echo $ofile
  echo $method_fmt

  # apply max ent with newton's method
  ~/TICG-chromatin/maxent/bin/run.sh $ofile $gamma $trust_region $mode $productionSweeps $equilibSweeps $goalSpecified $3 $numIterationsCopy $overwrite $1 $finalSimProductionSweeps

  # run.sh moves all data to $ofile upon completion
  cd $ofile

  # compare results
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --k $k --replicate_folder $ofile --save_npy > contact.log
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${ofile}/y.npy" --y_diag "$sampleFolder/y_diag.npy" --yhat_diag "${ofile}/y_diag.npy" >> contact.log

  echo ''
}

max_ent() {
  param_setup
  format_method
  for j in 1
  do
    scratchDirI="${scratchDir}/TICG_maxent${i}"
    mkdir -p $scratchDirI
    cd $scratchDirI
    max_ent_inner $scratchDirI $j > bash.log &
    i=$(( $i + 1 ))
  done
}

max_ent_inner () {
  # args:
  # 1 = scratchDir
  # 2 = replicate index
  ofile="${sampleFolder}/${method_fmt}/k${k}/replicate${2}"
  echo $ofile

  # move to scratch
  scratchDirResources="${1}/resources"
  mkdir -p $scratchDirResources

  cd $resources
  init_config="input${m}.xyz"
  echo $init_config
  if [ -f $init_config ]
  then
    echo 'here'
    cp $init_config $scratchDirResources
  else
    init_config='none'
  fi

  cd $scratchDirResources
  # generate sequences
  echo "starting get_params"
  echo $method_fmt
  python3 ~/TICG-chromatin/scripts/get_params.py --method=$method_fmt --m $m --k $k --sample $sample --data_folder $dataFolder --plot --epigenetic_data_folder $epiData --ChromHMM_data_file $chromHMMData --gnn_model_path $GNNModelPath --mlp_model_path $MLPModelPath --seq_seed $seqSeed --chi_method 'random' --min_chi=-1 --max_chi=1 --chi_seed $chiSeed --diag_chi_method $diagChiMethod --diag_bins $diagBins --diag_pseudobeads_on $diagPseudobeadsOn > params.log

  # get config
  echo "starting get_config"
  python3 ~/TICG-chromatin/scripts/get_config.py --m $m --max_ent --default_config "${resources}/default_config_maxent.json" --use_ematrix $useE --use_smatrix $useS --use_ground_truth_chi $useGroundTruthChi --use_ground_truth_diag_chi $useGroundTruthDiagChi --use_ground_truth_TICG_seed $useGroundTruthSeed --TICG_seed $TICGSeed --sample_folder $sampleFolder --load_configuration_filename $init_config > config.log


  # generate goals
  if [ $goalSpecifiedCopy = 'true' ]
  then
    echo "starting goal_specified"
    python3 ~/TICG-chromatin/maxent/bin/get_goal_experimental.py --m $m --contact_map "${sampleFolder}/y.npy" --mode $mode --diag_bins $diagBins > goal.log
  else
    echo "goal_specified is false"
  fi

  # apply max ent with newton's method
  ~/TICG-chromatin/maxent/bin/run.sh $ofile $gamma $trust_region $minDiagChi $mode $productionSweeps $equilibSweeps $goalSpecifiedCopy 1 $numIterationsCopy $overwrite $1 $finalSimProductionSweeps

  # run.sh moves all data to $ofile upon completion
  cd $ofile

  # compare results
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --k $k --replicate_folder $ofile --save_npy > contact.log
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${ofile}/y.npy" --y_diag "$sampleFolder/y_diag.npy" --yhat_diag "${ofile}/y_diag.npy" >> contact.log

  echo ''
}

format_method () {
  method_fmt=$method

  if [ $method = 'GNN' ]
  then
    method_fmt="${method_fmt}-${GNNModelID}"
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

  if [ $useGroundTruthDiagChi = 'true' ]
  then
    method_fmt="${method_fmt}-diag_chi"
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

  # diag mlp
  if [ $diagChiMethod = 'mlp' ]
  then
    method_fmt="${method_fmt}-diagMLP-${MLPModelID}"
  fi

  echo $method_fmt
}

param_setup(){
  numIterationsCopy=$numIterations
  goalSpecifiedCopy=$goalSpecified
  if [ $useS = 'true' ] || [ $useE = 'true' ]
  then
    useGroundTruthChi='false' # defaults to false anyways
    if ! [ $mode = 'diag' ]
    then
      numIterationsCopy=0
      goalSpecifiedCopy='false'
    fi
    if ! [ $loadChi = 'true' ]
    then
      k=0
    fi
  fi

  if [ $useGroundTruthChi == 'true' ] && ! [ $mode = 'plaid' ]
  then
    numIterationsCopy=0
    goalSpecifiedCopy='false'
  fi

  if [ $mode = 'plaid' ]
  then
    useGroundTruthDiagChi='true'
  fi

  if [ $useGroundTruthDiagChi = 'true' ]
  then
    diagChiMethod='none'
  fi

  dataFolder="${dir}/${dataset}"
  GNNModelPath="${results}/${GNNModelType}/${GNNModelID}"
  MLPModelPath="${results}/MLP/${MLPModelID}"
  sampleFolder="${dataFolder}/samples/sample${sample}"
}
