#! /bin/bash

# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin
#
# echo 'generate_params'
# dataset="dataset_11_21_22"
# python  ~/TICG-chromatin/bin/generate_params.py --samples 2400 --k 8 --dataset $dataset
# mv "~/${dataset}" /project2/depablo/erschultz



for i in 1
# 2 3 4 5 6 7 8
do
  echo $i
  sbatch ~/TICG-chromatin/bin/diag_dataset2/diag_dataset${i}.sh
done
