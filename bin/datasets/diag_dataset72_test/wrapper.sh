#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_11_23_23_test"
echo "generate_params for ${dataset}"
# uses poly12 fit to meandistS params + grid
# python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 10 --b 180 --v 8 --k 10 --ar 1.5 --m 512 --dataset $dataset --exp_dataset 'dataset_11_20_23' --cell_line 'imr90' --seq_mode 'eig_norm' --diag_mode 'meanDist_S_grid_poly8_log_start5' --data_dir '/home/erschultz'

# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup

sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset72_test/diag_dataset_fns.sh
start=1
end=10
for i in 1
do
  echo $i $start $end
  bash ~/TICG-chromatin/bin/datasets/bash_files/diag_dataset${i}.sh $sourceFile $start $end 10 "72_"
done
