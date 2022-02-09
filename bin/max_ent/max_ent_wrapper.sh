#! /bin/bash


for i in 1 2 3 4 13 14
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
