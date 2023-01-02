#! /bin/bash

# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_01_03_23"
echo "generate_params for ${dataset}"
# this dataset uses shuffled experimental PCs as sequences, with wider range of chi values, and k = 12
python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 5000 --k 12 --dataset $dataset --seq_mode 'pcs_shuffle' --chi_param_version 'v2'
# mv "~/${dataset}" /project2/depablo/erschultz

# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup
#
# sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset7/diag_dataset_fns.sh
# for i in 1 2 3 4 5 6 7 8 9 10
# do
#   echo $i
#   sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile
# done
