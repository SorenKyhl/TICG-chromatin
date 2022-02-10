#! /bin/bash

cd ~/TICG-chromatin/src
make
mv TICG-engine ..

for i in 1 2 3 4 13 14
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
