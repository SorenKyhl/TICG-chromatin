#! /bin/bash


for i in 1 2 3 4 5 6 7 8
do
  sbatch ~/TICG-chromatin/bin/max_ent${i}.sh
done
