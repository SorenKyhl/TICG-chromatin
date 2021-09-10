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
scratchDir=$10
method=$11

sampleFolder="$dataFolder/samples/sample$sample"

# other parameters
mode="plaid"
gamma=0.00001
gammaDiag=0.00001
resources="~/TICG-chromatin/maxent/resources"

# move to scratch
scratchDirResources="${scratchDir}/resources"
mkdir -p $scratchDirResources
cd $scratchDirResources

printf "\n${method} k=${k}\n"

# get config
python3 ~/TICG-chromatin/scripts/get_config.py --k $k --m $m --min_chi=-1 --max_chi=1 --save_chi_for_max_ent --goal_specified $goalSpecified --default_config "${resources}/default_config.json"

# generate sequences
python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --k $k --sample $sample --data_folder $dataFolder --plot

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
