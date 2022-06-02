#! /bin/bash

# module unload gcc
# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 1
# 10 11 13 14 -running
# 4 - done
# 3 - look closely here
# 12 - obscure error due to size of array
# GNN doesnt work for sample 13 or larger in 5_18 - insufficent RAM
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
