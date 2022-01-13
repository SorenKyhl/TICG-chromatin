#! /bin/bash


for i in 9 10 11 12 13 14
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent${i}.sh
done
