#! /bin/bash
#SBATCH --job-name=contact
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/erschultz/sequences_to_contact_maps/dataset_soren'

source activate python3.9_pytorch1.9


for i in 2
do
  replicateFolder="${dataFolder}/samples/sample${i}/me_seed0"
  python3 ~/TICG-chromatin/scripts/contact_map.py --replicate_folder $replicateFolder --save_npy &
done

wait
