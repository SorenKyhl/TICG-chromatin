#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=2000

cd /project2/depablo/erschultz/dataset_12_11_21/samples
cd sample40
rm -r ground_truth* &

cd ../sample1230
rm -r ground_truth* &

cd ../sample1718
rm -r ground_truth* &

cd /project2/depablo/erschultz/dataset_12_12_21/samples
cd sample40
rm -r ground_truth* GNN* PCA* kPCA* nmf* random* k_means* &

cd ../sample1230
rm -r ground_truth* GNN* PCA* kPCA* nmf* random* k_means* &

cd ../sample1718
rm -r ground_truth* GNN* PCA* kPCA* nmf* random* k_means* &

cd /project2/depablo/erschultz/dataset_10_27_21/samples
cd sample40
rm -r ground_truth* &

cd ../sample1230
rm -r ground_truth* &

cd ../sample1718
rm -r ground_truth* &
