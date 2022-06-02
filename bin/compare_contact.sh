#! /bin/bash
#SBATCH --job-name=compare_contact
#SBATCH --output=logFiles/compare_contact.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=2000


dataFolder='/project2/depablo/erschultz/dataset_05_18_22'
dataFolder='/home/erschultz/sequences_to_contact_maps/dataset_04_27_22'
dir='/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1'
hatdir='/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1/PCA-MSE/k10'

source activate python3.9_pytorch1.9
source activate python3.9_pytorch1.9_cuda10.2

STARTTIME=$(date +%s)

for sample in $(seq 1 1)
do
  echo $sample
  # python3 ~/TICG-chromatin/scripts/compare_contact.py --y "${dir}/y.npy" --y_diag "${dir}/y_diag.npy" --yhat "${hatdir}/y.npy" --yhat_diag "${hatdir}/y_diag.npy" --m=-1 --dir $hatdir
  python3 ~/TICG-chromatin/scripts/compare_contact_post.py --data_folder $dataFolder --sample $sample
done

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
