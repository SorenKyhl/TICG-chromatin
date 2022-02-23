#! /bin/bash
#SBATCH --job-name=contact
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/eric/sequences_to_contact_maps/dataset_11_14_21'
m=1500
finalIt=1

source activate python3.8_pytorch1.8.1_cuda11.1


for sample in 40
do
  replicateFolder="${dataFolder}/samples/sample${sample}/ground_truth-rank4-E/knone/replicate1/ground_truth-E/knone/replicate1"
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --replicate_folder $replicateFolder --save_npy --final_it $finalIt
done
