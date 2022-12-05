#! /bin/bash
#SBATCH --job-name=contact
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/erschultz/dataset_11_14_22'

source activate python3.9_pytorch1.9


for i in 201
do
  replicateFolder="${dataFolder}/samples/sample${i}/GNN-254-E/k0/replicate1"
  ofile="${replicateFolder}/contact.log"
  python3 ~/TICG-chromatin/scripts/contact_map.py --replicate_folder $replicateFolder --save_npy --plot > $ofile &
done

wait
