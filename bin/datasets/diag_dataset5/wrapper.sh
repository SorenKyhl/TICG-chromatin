#! /bin/bash

# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

echo 'generate_params'
dataset="dataset_12_20_22"
# this dataset uses experimental PCs as sequences, but a wider range of chi values than dataset_12_18_22
python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 5000 --k 8 --dataset $dataset --seq_mode 'pcs' --chi_param_version 'v2'
mv "~/${dataset}" /project2/depablo/erschultz

cd "/home/erschultz/${dataset}"
tar -czvf setup.tar.gz setup

# sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset5/diag_dataset_fns.sh
# for i in 1 2 3 4 5 6 7 8 9 10
# do
#   echo $i
#   sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile
# done
