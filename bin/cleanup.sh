#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd /project2/depablo/erschultz/dataset_10_27_21/samples

cd sample40
rm -r random &
rm -r PCA* &
rm -r kPCA* &
rm -r k_means &
rm -r ground* &
rm -r GNN* &
rm -r nmf* &

cd ../sample1230
rm -r random &
rm -r PCA* &
rm -r kPCA* &
rm -r k_means &
rm -r ground* &
rm -r GNN* &
rm -r nmf* &

cd ../sample1718
rm -r random &
rm -r PCA* &
rm -r kPCA* &
rm -r k_means &
rm -r ground* &
rm -r GNN* &
rm -r nmf* &

cd /project2/depablo/erschultz/dataset_11_03_21/samples

cd sample40
rm -r random &
rm -r PCA* &
rm -r kPCA* &
rm -r k_means &
rm -r ground* &
rm -r GNN* &
rm -r nmf* &

cd ../sample1230
rm -r random &
rm -r PCA* &
rm -r kPCA* &
rm -r k_means &
rm -r ground* &
rm -r GNN* &
rm -r nmf* &

cd ../sample1718
rm -r random &
rm -r PCA* &
rm -r kPCA* &
rm -r k_means &
rm -r ground* &
rm -r GNN* &
rm -r nmf* &
