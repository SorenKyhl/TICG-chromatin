#! /bin/bash


for i in 10 11
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent${i}.sh
done
