#! /bin/bash
#SBATCH --job-name=latex
#SBATCH --output=logFiles/latex.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000
#SBATCH --qos=depablo-debug

source activate python3.8_pytorch1.8.1_cuda10.2_2
dataDir='/project2/depablo/erschultz'
# dataDir='/home/erschultz/sequences_to_contact_maps'


dataset='dataset_01_17_22'
samples='1-2-3-4-5'
dataFolder="${dataDir}/${dataset}"
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
#

for sample in 1 2 3 4 5
do
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample
done
