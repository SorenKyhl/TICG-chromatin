#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=6:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir=/project2/depablo/erschultz/dataset_09_26_22/samples

for i in $(seq 1 2520)
do
  cd "${dir}/sample${i}"
  rm D.npy
done
