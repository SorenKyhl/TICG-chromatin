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

# dir='/home/erschultz/dataset_04_05_23/samples'
# cd $dir
#
# for i in {1001..1210}
# do
#   echo $i
#   for j in {0..5}
#   do
#     cd  "${dir}/sample${i}/optimize_grid_b_140_phi_0.03-max_ent10/iteration${j}"
#     # tar -czf equilibration.tar.gz equilibration
#     # rm -r equilibration
#     tar -czf production_out.tar.gz production_out
#     rm -r production_out
#   done
# done

dir='/home/erschultz/timing_analysis/2560/samples'
cd $dir

for i in {1..15}
do
  cd  "${dir}/sample${i}"
  pwd
  rm -r optimize_grid_b_63_phi_0.03-max_ent10

done
