#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_04_05_24_imr90"
echo "generate_params for ${dataset}"
# python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 10000 --b 200 --v 8 --k 10 --ar 1.5 --m 512 --dataset $dataset --exp_dataset 'dataset_12_06_23' --cell_line 'imr90' --seq_mode 'eig_norm' --diag_mode 'max_ent_grid_start2' --plaid_mode 'KDE' --data_dir '/project2/depablo/erschultz'

# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup

sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset89_imr90/diag_dataset_fns.sh
start=1
end=1000
for i in {1..10}
do
  echo $i $start $end
  sbatch ~/TICG-chromatin/bin/datasets/bash_files/diag_dataset${i}.sh $sourceFile $start $end 128 "89_"
  start=$(( $start + 1000 ))
  end=$(( $end + 1000 ))
done
