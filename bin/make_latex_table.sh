#! /bin/bash
#SBATCH --job-name=latex
#SBATCH --output=logFiles/latex.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

source activate python3.8_pytorch1.8.1_cuda10.2
small='true'

samples=1
dataFolder='/project2/depablo/erschultz/dataset_09_21_21'
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small $small

samples="40-1230-1718"
dataFolder='/project2/depablo/erschultz/dataset_08_29_21'
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small $small
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

samples="1201-1202-1203-40-1230-1718"
dataFolder='/project2/depablo/erschultz/dataset_08_26_21'
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small $small
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

samples="1201-1202-1203-40-1230-1718"
dataFolder='/project2/depablo/erschultz/dataset_08_24_21'
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --small $small
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
