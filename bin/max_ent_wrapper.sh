#! /bin/bash


for i in 1 3 4 5 6 7 8 9 10
do
  sbatch ~/TICG-chromatin/bin/max_ent${i}.sh
done
