#! /bin/bash
#SBATCH --job-name=contact
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivybw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/erschultz/dataset_02_04_23'

source activate python3.9_pytorch1.9


for i in 201
do
  folder="/home/erschultz/dataset_test/samples/sample5002/PCA-normalize-scale-S-b_177_phi_0.06/k8/replicate1/iteration1"
  ofile="${folder}/contact.log"
  python3 ~/TICG-chromatin/scripts/contact_map.py --sample_folder $folder --save_npy --plot --random_mode > $ofile &
done

wait
