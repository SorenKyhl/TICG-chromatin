#! /bin/bash
#SBATCH --job-name=wrapper
#SBATCH --output=logFiles/wrapper.out
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

dataset="dataset_08_25_23"
echo "generate_params for ${dataset}"
# uses poly6 fit to max ent params + grid
python  ~/TICG-chromatin/bin/datasets/generate_params.py --samples 5000 --b 261 --phi 0.01 --k 10 --m 512 --dataset $dataset --seq_mode 'eig_norm' --plaid_mode 'KDE' --diag_mode 'meanDist_S_grid' --data_dir '/home/erschultz'
#
cd "/home/erschultz/${dataset}"
tar -czvf setup.tar.gz setup
#
# cd "/project2/depablo/erschultz/${dataset}"
# tar -xzf setup.tar.gz
# rm -r samples
# #
# sourceFile=~/TICG-chromatin/bin/datasets/diag_dataset45/diag_dataset_fns.sh
# for i in 1 2 3 4 5 6 7 8 9 10
# do
#   echo $i
#   sbatch ~/TICG-chromatin/bin/datasets/diag_dataset${i}.sh $sourceFile
# done
