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
# python  ~/TICG-chromatin/bin/diag_dataset3/generate_params.py
# mv ~/dataset_12_05_22 /project2/depablo/erschultz


for i in 1 2 3 4 5 6 7 8 9 10
do
  echo $i
  sbatch ~/TICG-chromatin/bin/diag_dataset3/diag_dataset${i}.sh &
done

wait
