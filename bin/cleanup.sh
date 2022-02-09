#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir=/home/eric/sequences_to_contact_maps/dataset_09_21_21/samples_small

for i in 6 7 8 9 10 11 12 13 14 15 17 18 19 20 21 23 24 25
do
  cd "${dir}/sample${i}"
  rm -r PCA_analysis
  rm *.png
done
