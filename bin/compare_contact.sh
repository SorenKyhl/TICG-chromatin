#! /bin/bash
#SBATCH --job-name=compare_contact
#SBATCH --output=logFiles/compare_contact.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


dataFolder='/home/eric/dataset_test'
sample=83

# source activate python3.8_pytorch1.8.1_cuda10.2


sampleFolder="${dataFolder}/samples/sample${sample}/PCA"

# for k in 1 2 4 6
# do
#   cd "${sampleFolder}/k${k}/replicate1/iteration101"
#   python3 ~/TICG-chromatin/scripts/contact_map.py --save_npy --ifile 'production_out/contacts.txt'
# done


for sample in 82 83
do
  python3 ~/TICG-chromatin/scripts/compare_contact_post.py --data_folder $dataFolder --sample $sample
  # python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --y "$sampleFolder/y.npy" --yhat "${ofile}/iteration${prodIt}/y.npy" --y_diag "$sampleFolder/y_diag.npy" --yhat_diag "${ofile}/iteration${prodIt}/y_diag.npy"
done
