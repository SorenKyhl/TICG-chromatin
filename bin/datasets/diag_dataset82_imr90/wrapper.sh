#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_02_14_24_imr90"
echo "generate_params for ${dataset}"
# python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 5000 --b 200 --v 8 --k 10 --ar 1.5 --m 512 --dataset $dataset --exp_dataset 'dataset_12_06_23' --cell_line 'imr90' --seq_mode 'eig_norm' --diag_mode 'meanDist_S_grid_poly8_log_start1' --plaid_mode 'KDE' --data_dir '/project/depablo/erschultz'

# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup

sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset82_imr90/diag_dataset_fns.sh
start=1000
end=1500
for i in {6..7}
do
  echo $i $start $end
  # sbatch ~/TICG-chromatin/bin/datasets/bash_files/diag_dataset${i}.sh $sourceFile $start $end 128 "82_"
  start=$(( $start + 500 ))
  end=$(( $end + 500 ))
done
