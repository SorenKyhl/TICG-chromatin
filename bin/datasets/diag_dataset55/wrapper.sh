#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_09_26_23"
echo "generate_params for ${dataset}"
# uses poly6 fit to max ent params + grid
# python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 10000 --b 180 --phi 0.008 --k 10 --ar 1.5 --m 512 --dataset $dataset --seq_mode 'eig_norm' --diag_mode 'max_ent_poly6_log_v2_grid' --data_dir '/project2/depablo/erschultz'
#
cd "/home/erschultz/${dataset}"
tar -czvf setup.tar.gz setup
#
# cd "/project2/depablo/erschultz/${dataset}"
# tar -xzf setup.tar.gz
# rm -r samples
#
sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset55/diag_dataset_fns.sh
for i in {11..20}
do
  echo $i
  sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile 24
done

for i in {21..25}
do
  echo $i
  sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile 128
done
