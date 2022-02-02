#! /bin/bash
#SBATCH --job-name=contact
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/eric/dataset_test'
k=3
m=1500

source activate python3.8_pytorch1.8.1_cuda10.2


for sample in 80 81 82 83 84 85 86
do
  dir="${dataFolder}/samples/sample${sample}"
  ifile="${dir}/data_out/contacts.txt"
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --ifile $ifile --odir $dir --save_npy
done
