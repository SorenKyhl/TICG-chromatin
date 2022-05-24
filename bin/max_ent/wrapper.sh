#! /bin/bash

# module unload gcc
# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 3
# 1 4 5 6 7 8 11 12 13 14 9 - running
# 7 9 10 11 12 - only ran part of it
# 2 - worked see latex
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
