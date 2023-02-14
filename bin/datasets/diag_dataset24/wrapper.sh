#! /bin/bash
#SBATCH --job-name=wrapper7
#SBATCH --output=logFiles/wrapper7.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu

# module load gcc/10.2.0
# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.8_pytorch1.9

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

dataset="dataset_02_14_23"
echo "generate_params for ${dataset}"
# python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 5000 --k 8 --m 512 --dataset $dataset --seq_mode 'eig_norm' --chi_param_version 'v12' --diag_mode 'linear'
#
# cd "/home/erschultz/${dataset}"
# tar -czvf setup.tar.gz setup
#
# cd /project2/depablo/erschultz/dataset_01_06_23
# tar -xzf setup.tar.gz
# rm -r samples

sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset24/diag_dataset_fns.sh
for i in 1 2 3 4 5 6 7 8 9 10
do
  echo $i
  sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile
done
