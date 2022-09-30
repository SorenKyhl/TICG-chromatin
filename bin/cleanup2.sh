#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --output=logFiles/cleanup2.out
#SBATCH --time=2:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir=/project2/depablo/erschultz/dataset_05_18_22/samples

rm -r "${dir}/sample16"
rm -r "${dir}/sample17"
rm -r "${dir}/sample18"

for i in $(seq 1 15)
do
  cd "${dir}/sample${i}"
  rm *.png
  rm *.txt
  rm -r PCA-normalize-diagMLP*
  rm -r GNN-150-E
done

dir=/project2/depablo/erschultz/dataset_11_03_21/samples
for i in $(seq 1 2000)
do
  cd "${dir}/sample${i}"
  rm -r data_out

done
