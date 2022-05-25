#! /bin/bash

# module unload gcc
# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 3 12
# 5 6 7 10 13 14 - running
# 1 4 5 - done
# 8 - think it worked - check latex
# 9 11 - insufficent RAM
# 10 11 12 14 - only ran part of it
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
