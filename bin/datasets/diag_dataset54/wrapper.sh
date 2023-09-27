#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_09_25_23"
echo "generate_params for ${dataset}"
# uses poly6 fit to max ent params + grid
# python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 10000 --b 180 --phi 0.008 --k 5 --ar 1.5 --m 512 --dataset $dataset --seq_mode 'eig_norm' --diag_mode 'max_ent_poly6_log_v2_grid' --data_dir '/project2/depablo/erschultz'
#
# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup
#
# cd "/project2/depablo/erschultz/${dataset}"
# tar -xzf setup.tar.gz
# rm -r samples

sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset54/diag_dataset_fns.sh
start=1
end=400
for i in {1..4}
do
  echo $i $start $end
  sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile $start $end 24 "54_"
  start=$(( $start + 400 ))
  end=$(( $end + 400 ))
done

start=2001
end=2400
for i in {6..9}
do
  echo $i $start $end
  sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile $start $end 24 "54_"
  start=$(( $start + 400 ))
  end=$(( $end + 400 ))
done

start=4001
end=4400
for i in {11..14}
do
  echo $i $start $end
  sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile $start $end 24 "54_"
  start=$(( $start + 400 ))
  end=$(( $end + 400 ))
done

start=7201
end=7600
i=19
echo $i $start $end
sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile $start $end 24 "54_"


start=8000
end=8400
i=20
echo $i $start $end
sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile $start $end 24 "54_"


start=8800
end=9600
i=5
echo $i $start $end
sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile $start $end 24 "54_"


sleep 100
