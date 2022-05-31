#! /bin/bash
#SBATCH --job-name=latex4
#SBATCH --output=logFiles/latex4.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=5
#SBATCH --mem=0
#SBATCH --qos=depablo-debug

local='false'
if [ $local = 'true' ]
then
  dataDir='/home/erschultz/sequences_to_contact_maps'
  source activate python3.9_pytorch1.9
else
  dataDir='/project2/depablo/erschultz'
  source activate python3.9_pytorch1.9_cuda10.2
fi

# dataSet='dataset_08_24_21'
# dataSet='dataset_08_26_21'
# dataSet='dataset_08_29_21'
# dataSet='dataset_10_27_21'
# dataSet='dataset_11_03_21'
# dataSet='dataset_11_14_21'

dataset='dataset_05_18_22'
dataFolder="${dataDir}/${dataset}"

for sample in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
 # 16 17 18 19
do
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample &
done

wait
