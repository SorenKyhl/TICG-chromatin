#! /bin/bash

source activate python3.9_pytorch1.9_cuda10.2
module unload gcc
module load gcc/10.2.0


cd ~/TICG-chromatin/src
make
mv TICG-engine ..

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

for i in 4
do
  echo $i
  sbatch ~/TICG-chromatin/bin/max_ent/max_ent${i}.sh
done
