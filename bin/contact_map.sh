#! /bin/bash
#SBATCH --job-name=contact
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/erschultz/dataset_test/'
m=200
finalIt=101

source activate python3.8_pytorch1.8.1_cuda11.1


for i in 1 2 4
do
  replicateFolder="${dataFolder}/samples/sample10/PCA/k${i}/replicate1/"
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --replicate_folder $replicateFolder --save_npy --final_it $finalIt --k $i
done
