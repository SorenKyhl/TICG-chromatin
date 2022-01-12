#! /bin/bash


for i in 6 7 8 9
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent${i}.sh
done
