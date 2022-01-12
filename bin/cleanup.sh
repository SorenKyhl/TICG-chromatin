#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd ~/project2/depablo/erschultz/dataset_01_11_22

cd sample40
cd ground_truth-psi-chi
rm -r k10

cd ../PCA
rm -r k6

cd ../../sample41
cd ground_truth-psi-chi
rm -r k10

cd ../PCA
rm -r k6

cd ../../sample42
cd ground_truth-psi-chi
rm -r k10

cd ../PCA
rm -r k6

cd ../../sample43
cd ground_truth-psi-chi
rm -r k10

cd ../PCA
rm -r k6

cd ../../sample44
cd ground_truth-psi-chi
rm -r k10

cd ../PCA
rm -r k6
