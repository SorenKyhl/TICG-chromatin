#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir=/project2/depablo/erschultz/dataset_09_21_21/samples

for i in 1 2 8 14 20
do
  cd "${dir}/sample${i}"
  cd GNN-137-E-diagOn
  rm -r k
  cd ../GNN-109-E-diagOn
  rm -r k
done
