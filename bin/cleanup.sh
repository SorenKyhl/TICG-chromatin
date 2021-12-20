#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd /project2/depablo/erschultz/dataset_12_11_21/samples
cd sample40
rm -r ground_truth*

cd ../sample1230
rm -r ground_truth*

cd ../sample1718
rm -r ground_truth*


cd /project2/depablo/erschultz/dataset_10_27_21/samples
cd sample40
rm -r ground_truth*

cd ../sample1230
rm -r ground_truth*

cd ../sample1718
rm -r ground_truth*

cd /project2/depablo/erschultz/dataset_12_17_21/samples
cd sample40
rm -r ground_truth* &
rm -r GNN* &
rm -r PCA* &
rm -r kPCA* &
rm -r nmf* &
rm -r random* &
rm -r k_means* &
wait

cd ../sample1230
rm -r ground_truth* &
rm -r GNN* &
rm -r PCA* &
rm -r kPCA* &
rm -r nmf* &
rm -r random* &
rm -r k_means* &
wait

cd ../sample1718
rm -r ground_truth* &
rm -r GNN* &
rm -r PCA* &
rm -r kPCA* &
rm -r nmf* &
rm -r random* &
rm -r k_means* &
wait

cd /project2/depablo/erschultz/dataset_12_12_21/samples
cd sample40
rm -r ground_truth* &
rm -r GNN* &
rm -r PCA* &
rm -r kPCA* &
rm -r nmf* &
rm -r random* &
rm -r k_means* &
wait

cd ../sample1230
rm -r ground_truth* &
rm -r GNN* &
rm -r PCA* &
rm -r kPCA* &
rm -r nmf* &
rm -r random* &
rm -r k_means* &
wait

cd ../sample1718
rm -r ground_truth* &
rm -r GNN* &
rm -r PCA* &
rm -r kPCA* &
rm -r nmf* &
rm -r random* &
rm -r k_means* &
wait
