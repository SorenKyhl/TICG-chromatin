#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_11_16_23_k562"
echo "generate_params for ${dataset}"
# uses poly12 fit to meandistS params + grid
# python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 5000 --b 180 --v 8 --k 10 --ar 1.5 --m 512 --dataset $dataset --exp_dataset 'dataset_06_29_23' --cell_line 'k562' --seq_mode 'eig_norm' --diag_mode 'meanDist_S_grid_poly8_log_start5' --data_dir '/project2/depablo/erschultz'

# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup

sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset69_k562/diag_dataset_fns.sh
start=1
end=1000
for i in {11..20}
do
  echo $i $start $end
  sbatch ~/TICG-chromatin/bin/datasets/bash_files/diag_dataset${i}.sh $sourceFile $start $end 48 "66_"
  start=$(( $start + 1000 ))
  end=$(( $end + 1000 ))
done
