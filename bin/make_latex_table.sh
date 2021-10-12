#! /bin/bash
#SBATCH --job-name=latex
#SBATCH --output=logFiles/latex.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

samples=1
dataFolder='/project2/depablo/erschultz/dataset_09_21_21'
small='true'

source activate python3.8_pytorch1.8.1_cuda10.2

python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small $small
