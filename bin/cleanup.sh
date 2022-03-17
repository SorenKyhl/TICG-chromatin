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
  rm -r kPCA-x* &
  rm -r kPCA-y* &
  rm -r GNN* &
  rm -r ground* &
  rm -r PCA_split* &
  rm -r nmf* &
  rm -r PCA* &
  rm -r k_means* &
  rm -r ChromHMM* &
  rm -r epigeneitc* &
  rm -r random* &
  wait
done
