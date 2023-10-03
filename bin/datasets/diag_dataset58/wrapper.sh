#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_09_29_23_test"
echo "generate_params for ${dataset}"
# uses poly6 fit to max ent params + grid
python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 3 --b 180 --phi 0.008 --k 10 --ar 1.5 --m 512 --dataset $dataset --seq_mode 'eig_norm' --diag_mode 'meanDist_S_grid' --data_dir '/home/erschultz'
#
# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup
#
# cd "/project2/depablo/erschultz/${dataset}"
# tar -xzf setup.tar.gz
# rm -r samples
#
sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset58/diag_dataset_fns.sh
start=1
end=1
for i in 1
do
  echo $i $start $end
  bash ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile $start $end 1 "58_"

done
