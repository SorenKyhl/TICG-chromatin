#! /bin/bash

# module unload gcc
# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 1
# 4 5 6 7 8 9 10 11 - running
# 1 4 - partially/maybe worked, re running
# 2 3 - worked see latex
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
