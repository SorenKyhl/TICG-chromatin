#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=6:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu

dir='/home/erschultz/dataset_02_04_23/samples'
cd $dir

for i in {211..283}
do
  mv "${dir}/sample${i}/optimize_grid_b_140_phi_0.03-max_ent" "${dir}/sample${i}/optimize_grid_b_140_phi_0.03-max_ent10"
done
