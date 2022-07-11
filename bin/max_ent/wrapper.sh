#! /bin/bash

# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 1 2 7 8 9
 # 3 4 - done
 # 5 - rerunning with more sweeps per iteration
 # 6 - GNN didn't work idk why - rerunning
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
