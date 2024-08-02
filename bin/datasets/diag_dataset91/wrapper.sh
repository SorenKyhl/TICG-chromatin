#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

start_i=1
end_i=5
for cell_line in 'gm12878', 'imr90', 'hap1', 'huvec', 'hmec'
do
    dataset="dataset_08_02_24_${cell_line}"
    sourceFile="~/TICG-chromatin/bin/datasets/diag_dataset91/${cell_line}.sh"
    start=1
    end=2000
    for i in $(seq $start_i $end_i)
    do
      echo $i $start $end
      sbatch ~/TICG-chromatin/bin/datasets/bash_files/diag_dataset${i}.sh $sourceFile $start $end 128 "${cell_line}_"
      start=$(( $start + 2000 ))
      end=$(( $end + 2000 ))
    done
    start_i=$(( $start_i + 5 ))
    end_i=$(( $end_i + 5 ))
done
