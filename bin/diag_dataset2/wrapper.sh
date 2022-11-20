#! /bin/bash

# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..



# necessary to ensure log files are in right place
cd ~/TICG-chromatin

python  ~/TICG-chromatin/bin/diag_dataset2/generate_params.py
mv ~/dataset_11_18_22 /project2/depablo/erschultz


for i in 1 2 3 4 5 6 7 8
do
  echo $i
  sbatch ~/TICG-chromatin/bin/diag_dataset2/diag_dataset${i}.sh &
done

wait
