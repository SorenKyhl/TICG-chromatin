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
rm -r PCA &
rm -r PCA_split &
rm -r kPCA* &
rm -r k_means &
rm -r ground_truth* &
rm -r GNN* &
rm -r nmf* &

wait

cd ../sample1230
rm -r random &
rm -r PCA &
rm -r PCA_split &
rm -r kPCA* &
rm -r k_means &
rm -r ground* &
rm -r GNN* &
rm -r nmf* &

wait

cd ../sample1718
rm -r random &
rm -r PCA &
rm -r PCA_split &
rm -r kPCA* &
rm -r k_means &
rm -r ground* &
rm -r GNN* &
rm -r nmf* &

wait
