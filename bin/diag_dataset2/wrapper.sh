#! /bin/bash

# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 1 2 3 4 5 6
do
  echo $i
  sbatch ~/TICG-chromatin/bin/diag_dataset2/diag_dataset${i}.sh &
done

wait
