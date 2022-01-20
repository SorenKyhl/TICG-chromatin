#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd /project2/depablo/erschultz/dataset_01_15_22/samples

cd sample40/k_means
rm -r k1

cd ../../sample1230/k_means
rm -r k1

cd ../../sample1718/k_means
rm -r k1

cd ../../sample1751/k_means
rm -r k1

cd ../../sample1761/k_means
rm -r k1
