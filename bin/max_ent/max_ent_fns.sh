#! /bin/bash
source activate python3.9_pytorch1.9
module load gcc/10.2.0

# directories
resources=~/TICG-chromatin/utils
results=~/sequences_to_contact_maps/results
dir='/project2/depablo/erschultz'
scratchDir='/home/erschultz/scratch'

# sweep params
productionSweeps=500000
dumpFrequency=50000
finalSimProductionSweeps=500000
equilibSweeps=50000
numIterations=12 # iteration 1 + numIterations is production run to get contact map
parallel='false'
numThreads=1

# energy params
useL='true'
useS='true'
useD='true'

# general params
overwrite=1
replicate=1
loadChi='false'
project='false'
goalSpecified=1
GNNModelType='ContactGNNEnergy'
m=-1
k=-1
seqSeed='none'
chiSeed='none'
TICGSeed='none'
bondType='gaussian'
chiMethod='random'
phiChromatin=0.06
boundaryType='spherical'


gridSize=28.7
beadVol=520
constantChi=0

# diag params
diagChiMethod='none'
dense='false'
diagBins=32
maxDiagChi=0
chiDiagSlope=1
chiDiagScale='none'
denseCutoff='none'
denseLoading='none'
smallBinSize=0
bigBinSize=-1
nSmallBins=0
nBigBins=-1
diagStart=0
diagCutoff='none'

# ground truth params
useGroundTruthChi='false'
useGroundTruthSeed='false'

# newton's method params
mode="plaid"
trust_region=50
gamma=1.

# experimental data
chromHMMData="${chipSeqFolder}/aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt"
start='none'
end='none'
chrom='none'
resolution=50000
cellLine=HCT116
dataType=fold_change_control

max_ent_resume(){
  # args:
  # 1 = start iteration
  param_setup
  format_method
  scratchDirI="${scratchDir}/TICG_maxent${i}"
  mkdir -p $scratchDirI
  cd $scratchDirI
  max_ent_resume_inner $scratchDirI $1 >> bash.log &
  i=$(( $i + 1 ))
}

max_ent_resume_inner(){
  # args:
  # 1 = scratchDir
  # 2 = start iteration
  odir="${sampleFolder}/${method_fmt}/k${k}/replicate1" # only replicate1 permitted
  echo $odir
  echo $method_fmt

  if [ -d "${odir}/resources"]
  then
    cp -r "${odir}/resources"
  fi

  # apply max ent with newton's method
  ~/TICG-chromatin/maxent/bin/run.sh -o $odir -d $1 -g $gamma -t $trust_region -c $mode -s $productionSweeps -e $equilibSweeps -z $goalSpecifiedCopy -q $2 -n $numIterationsCopy -w $overwrite -f $finalSimProductionSweeps

  # run.sh moves all data to $odir upon completion
  cd $odir

  # compare results
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --k $k --replicate_folder $odir --save_npy --plot > contact.log
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${odir}/y.npy" --y_diag "$sampleFolder/y_diag.npy" --yhat_diag "${odir}/y_diag.npy" >> contact.log

  echo ''
}

max_ent() {
  param_setup
  format_method

  scratchDirI="${scratchDir}/TICG_maxent${i}"
  mkdir -p $scratchDirI
  cd $scratchDirI
  max_ent_inner $scratchDirI $replicate > bash.log &
  i=$(( $i + 1 ))
}

max_ent_inner () {
  # args:
  # 1 = scratchDir
  # 2 = replicate index
  odir="${sampleFolder}/${method_fmt}/k${k}/replicate${2}"
  echo $odir

  # move to scratch
  scratchDirResources="${1}/resources"
  mkdir -p $scratchDirResources

  cd $resources
  init_config='none'

  cd $scratchDirResources
  # generate sequences
  echo "starting get_params"
  echo $method_fmt
  python3 ~/TICG-chromatin/scripts/get_params.py --config_ifile "${resources}/default_config_maxent.json" --method=$method_fmt --m $m --k $k --sample $sample --data_folder $dataFolder --plot --cell_line $cellLine --epi_data_type $dataType --resolution $resolution --start $start --end $end --chromosome $chrom --ChromHMM_data_file $chromHMMData --gnn_model_path $GNNModelPath --mlp_model_path $MLPModelPath --seq_seed $seqSeed --chi_method $chiMethod --min_chi=-1 --max_chi=1 --chi_seed $chiSeed --diag_chi_method $diagChiMethod --diag_bins $diagBins --max_diag_chi $maxDiagChi --dense_diagonal_on $dense --dense_diagonal_cutoff $denseCutoff --dense_diagonal_loading $denseLoading --small_binsize $smallBinSize --big_binsize $bigBinSize --n_small_bins $nSmallBins --n_big_bins $nBigBins --diag_start $diagStart --diag_cutoff $diagCutoff --diag_chi_slope $chiDiagSlope --diag_chi_scale $chiDiagScale > params.log

  echo "starting get_config"
  python3 ~/TICG-chromatin/scripts/get_config.py --dump_frequency $dumpFrequency --parallel $parallel --num_threads $numThreads --m $m --max_ent --mode $mode --bond_type $bondType --bond_length $bondLength --grid_size $gridSize --bead_vol $beadVol --dense_diagonal_on $dense --use_lmatrix $useL --use_smatrix $useS --use_dmatrix $useD --use_ground_truth_chi $useGroundTruthChi --use_ground_truth_TICG_seed $useGroundTruthSeed --TICG_seed $TICGSeed --sample_folder $sampleFolder --load_configuration_filename $init_config --phi_chromatin $phiChromatin --boundary_type $boundaryType --constant_chi $constantChi > config.log


  # generate goals
  if [ $goalSpecifiedCopy -eq 1 ]
  then
    echo "starting goal_specified"
    python3 ~/TICG-chromatin/maxent/bin/get_goal_experimental.py --contact_map "${sampleFolder}/y.npy" --mode $mode --verbose --grid_size $gridSize --v_bead $beadVol > goal.log
  else
    echo "goal_specified is false"
  fi

  # apply max ent with newton's method
  ~/TICG-chromatin/maxent/bin/run.sh -o $odir -d $1 -g $gamma -t $trust_region -c $mode -s $productionSweeps -e $equilibSweeps -z $goalSpecifiedCopy -n $numIterationsCopy -w $overwrite -f $finalSimProductionSweeps

  # run.sh moves all data to $odir upon completion
  cd $odir

  # compare results
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --k $k --replicate_folder $odir --save_npy --plot > contact.log
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${odir}/y.npy" --y_diag "$sampleFolder/y_diag.npy" --yhat_diag "${odir}/y_diag.npy" >> contact.log

  echo ''
}

format_method () {
  method_fmt=$method

  if [ $method = 'chromhmm' ]
  then
    method_fmt="${method_fmt}-${dataType}"
  fi

  if [ $method = 'GNN' ]
  then
    method_fmt="${method_fmt}-${GNNModelID}"
  fi

  if [ $method = 'GNN-kr' ] # TODO
  then
    method_fmt="${method_fmt}-${GNNModelID}"
  fi

  if [ $method = 'GNN-clean' ] # TODO
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

  if [ $diagChiMethod = 'ground_truth' ]
  then
    method_fmt="${method_fmt}-diag_chi"
  fi

  if [ $useS = 'true' ]
  then
    method_fmt="${method_fmt}-S"
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

  if [ $mode = 'diag' ]
  then
    method_fmt="${method_fmt}-diag"
  fi
}

param_setup(){
  numIterationsCopy=$numIterations
  goalSpecifiedCopy=$goalSpecified

  if [ $useGroundTruthChi == 'true' ] && ! [ $mode = 'plaid' ]
  then
    numIterationsCopy=0
    goalSpecifiedCopy=0
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
