#! /bin/bash


for i in 3 4 5 6 7 8 9 10 11 12
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent${i}.sh
done
