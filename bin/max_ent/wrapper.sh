#! /bin/bash

# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 3 4 5 6
 # 3 4 - mlp didnt work - fixed
 # 5 - rerunning with more sweeps per iteration
 # 6 - GNN didn't work idk why
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
