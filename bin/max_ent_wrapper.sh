#! /bin/bash


for i in 6 7 8
do
  sbatch ~/TICG-chromatin/bin/max_ent${i}.sh
done
