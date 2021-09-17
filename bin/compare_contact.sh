#! /bin/bash
#SBATCH --job-name=compare_contact
#SBATCH --output=compare_contact.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/project2/depablo/erschultz/dataset_08_24_21'
sample=1201

source activate python3.8_pytorch1.8.1_cuda10.2

python3 ~/TICG-chromatin/compare_contact_post.py --data_folder $dataFolder, --sample $sample
