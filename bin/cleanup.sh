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
  rm *.png
  rm y_diag.npy
  rm *.txt
done

dir=/project2/depablo/erschultz/dataset_04_27_22/samples

for i in $(seq 1 2000)
do
  cd "${dir}/sample${i}"
  rm *.png
  rm *.txt
done

dir=/project2/depablo/erschultz/dataset_05_12_22/samples

for i in $(seq 1 2000)
do
  cd "${dir}/sample${i}"
  rm *.png
  rm *.txt
  rm e.npy
done
