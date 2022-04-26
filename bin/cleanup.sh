#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir=/project2/depablo/erschultz/dataset_04_26_22/samples

for i in $( seq 2001 4400 )
do
  rm -r "sample${i}"
done
