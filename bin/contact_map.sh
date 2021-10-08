#! /bin/bash
#SBATCH --job-name=compare_contact
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/project2/depablo/erschultz/dataset_09_21_21'
sample=1
method='ChromHMM'
k=9
m=-1024
iteration=101
dir="${dataFolder}/samples/sample${sample}/${method}/k${k}/iteration${iteration}"
ifile="${dir}/production_out/contacts.txt"

source activate python3.8_pytorch1.8.1_cuda10.2

python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --ifile $ifile --odir $dir --save_npy
