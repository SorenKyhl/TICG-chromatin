#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd /project2/depablo/erschultz/dataset_09_21_21/samples/sample1

rm -r PCA &
rm -r PCA_split &

wait

cd epigenetic/k2
mkdir replicate1
mv * replicate1

cd ../k4
mkdir replicate1
mv * replicate1

cd ../k6
mkdir replicate1
mv * replicate1

cd ../k8
mkdir replicate1
mv * replicate1

cd ../k9
mkdir replicate1
mv * replicate1

cd ../../k_means/k2
mkdir replicate1
mv * replicate1

cd ../k4
mkdir replicate1
mv * replicate1

cd ../k6
mkdir replicate1
mv * replicate1

cd ../../nmf/k2
mkdir replicate1
mv * replicate1

cd ../k4
mkdir replicate1
mv * replicate1

cd ../k6
mkdir replicate1
mv * replicate1
