#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --output=logFiles/cleanup2.out
#SBATCH --time=2:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dataset=dataset_11_14_22
for i in 1 2 5 6 9 10 13 14 16 18
do
  dir="/home/erschultz/${dataset}/samples/sample${i}"
  max_ent_dir="${dir}/PCA_split-binarizeMean-E/k8/replicate1"
  cd $max_ent_dir
  rm *edit.png
done
