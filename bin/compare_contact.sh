#! /bin/bash
#SBATCH --job-name=compare_contact
#SBATCH --output=logFiles/compare_contact.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/erschultz/sequences_to_contact_maps/dataset_11_14_21'
sample=40
dir='/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1'
hatdir='/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1/PCA-MSE/k3'

source activate python3.8_pytorch1.8.1_cuda11.1

for sample in 40
do
  python3 ~/TICG-chromatin/scripts/compare_contact.py --y "${dir}/y.npy" --y_diag "${dir}/y_diag.npy" --yhat "${hatdir}/y.npy" --yhat_diag "${hatdir}/y_diag.npy" --m=-1 --dir $hatdir
  # python3 ~/TICG-chromatin/scripts/compare_contact_post.py --data_folder $dataFolder --sample_folder $sampleFolder
done
