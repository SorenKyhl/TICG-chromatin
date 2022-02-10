#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir=/home/eric/sequences_to_contact_maps/dataset_11_14_21/samples

for i in 40 1230 1718
do
  cd "${dir}/sample${i}"
  rm -r kPCA-x* &
  rm -r kPCA-y* &
  rm -r GNN* &
  rm -r ground* &
  rm -r PCA_split* &
  rm -r nmf &
  rm -r PCA &
  rm -r k_means &
  wait
done
