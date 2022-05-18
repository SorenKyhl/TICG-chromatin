#! /bin/bash
#SBATCH --job-name=latex2
#SBATCH --output=logFiles/latex2.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000
#SBATCH --qos=depablo-debug

local='false'
if [ $local = 'true' ]
then
  dataDir='/project2/depablo/erschultz'
  source activate python3.9_pytorch1.9
else
  dataDir='/home/erschultz/sequences_to_contact_maps'
  source activate python3.9_pytorch1.9_cuda10.2
fi


samples="40-1230-1718-1751-1761"
sample=40
# dataSet='dataset_08_24_21'
# dataSet='dataset_08_26_21'
# dataSet='dataset_08_29_21'
# dataSet='dataset_10_27_21'
# dataSet='dataset_11_03_21'
# dataSet='dataset_11_14_21'

dataset=dataset_04_27_22
samples="1-2-3-4"
dataFolder="${dataDir}/${dataset}"
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

for sample in 1 2 3 4
do
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample &
done

wait
