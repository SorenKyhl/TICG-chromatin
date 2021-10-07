#! /bin/bash

# command-line arguments
m=$1
k=$2
sample=$3
dataFolder=$4
productionSweeps=$5
equilibSweeps=$6
goalSpecified=$7
numIterations=$8
overwrite=$9
scratchDir=${10}
method=${11}
echo $@

sampleFolder="$dataFolder/samples/sample$sample"

# other parameters
mode="plaid"
gamma=0.00001
gammaDiag=0.00001
resources="/home/erschultz/TICG-chromatin/maxent/resources"
chipSeqFolder="/home/erschultz/sequences_to_contact_maps/chip_seq_data/"
epiData="${chipSeqFolder}/fold_change_control/processed"
chromHMMData="${chipSeqFolder}/aligned_reads/ChromHMM_15/STATEBYLINE/HTC116_15_chr2_statebyline.txt"

# move to scratch
scratchDirResources="${scratchDir}/resources"
mkdir -p $scratchDirResources
cd $scratchDirResources
cp "${resources}/input1024.xyz" .

# get config
python3 ~/TICG-chromatin/scripts/get_config.py --k $k --m $m --min_chi=-1 --max_chi=1 --save_chi_for_max_ent --goal_specified $goalSpecified --default_config "${resources}/default_config.json"

# generate sequences
python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --k $k --sample $sample --data_folder $dataFolder --plot --save_npy --epigenetic_data_folder $epiData --ChromHMM_data_file $chromHMMData

# generate goals
if [ $goalSpecified -eq 1 ]
then
  python3 ~/TICG-chromatin/maxent/bin/get_goal_experimental.py --m $m --k $k --contact_map "${sampleFolder}/y.npy"
fi

# apply max ent with newton's method
dir="${sampleFolder}/${method}/k${k}"
~/TICG-chromatin/maxent/bin/run.sh $dir $gamma $gammaDiag $mode $productionSweeps $equilibSweeps $goalSpecified $numIterations $overwrite $scratchDir

# compare results
prodIt=$(($numIterations+1))
cd $dir
python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${dir}/iteration${prodIt}/y.npy" --y_diag_instance "$sampleFolder/y_diag_instance.npy" --yhat_diag_instance "${dir}/iteration${prodIt}/y_diag_instance.npy"

echo "\n\n"
