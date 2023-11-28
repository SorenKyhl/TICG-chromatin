#! /bin/bash

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_11_27_23_imr90"
echo "generate_params for ${dataset}"
# uses poly12 fit to meandistS params + grid
# python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 5000 --b 180 --v 8 --k 10 --sim_k 5 --ar 1.5 --m 512 --dataset $dataset --exp_dataset 'dataset_11_20_23' --cell_line 'imr90' --seq_mode 'eig_norm' --diag_mode 'meanDist_S_grid_poly8_log_start1' --data_dir '/project2/depablo/erschultz'

# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup

sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset74_imr90/diag_dataset_fns.sh
start=1
end=1000
for i in {1..5}
do
  echo $i $start $end
  sbatch ~/TICG-chromatin/bin/datasets/bash_files/diag_dataset${i}.sh $sourceFile $start $end 128 "74_"
  start=$(( $start + 1000 ))
  end=$(( $end + 1000 ))
done
