#! /bin/bash

# module unload gcc
# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 1 2 6 13 14
# 3 5 7 10  - running
# 8 - think it worked - check latex
# 9 11 - insufficent RAM
# 12 - obscure error due to size of array
# 10 11 12 14 - only ran part of it
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
