#! /bin/bash
#SBATCH --job-name=contact
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/erschultz/dataset_02_04_23'

source activate python3.9_pytorch1.9


for i in 201
do
  replicateFolder="${dataFolder}/samples/sample${i}/none/k1/replicate1/samples/sample${i}_edit"
  ofile="${replicateFolder}/contact.log"
  python3 ~/TICG-chromatin/scripts/contact_map.py --replicate_folder $replicateFolder --save_npy --plot --random_mode > $ofile &
done

wait
