#! /bin/bash
#SBATCH --job-name=compare_contact
#SBATCH --output=logFiles/compare_contact.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/eric/sequences_to_contact_maps/dataset_11_14_21'
sample=40

source activate python3.8_pytorch1.8.1_cuda11.1

for sample in 40
do
  sampleFolder="${dataFolder}/samples/sample${sample}/ground_truth-rank3-E/knone/replicate1/"
  # python3 -m scripts.compare_contact_post --data_folder $dataFolder --sample_folder $sampleFolder
  python3 ~/TICG-chromatin/scripts/compare_contact_post.py --data_folder $dataFolder --sample_folder $sampleFolder
done
