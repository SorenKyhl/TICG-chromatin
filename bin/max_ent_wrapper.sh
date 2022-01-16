#! /bin/bash


for i in 1 2 3
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent${i}.sh
done
