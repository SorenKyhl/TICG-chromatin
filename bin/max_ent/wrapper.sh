#! /bin/bash

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 6 7 8 9 10 11
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
