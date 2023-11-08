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

# dir='/home/erschultz/dataset_02_04_23/samples'
# cd $dir
# for i in {201..282}
# do
#   cd  "${dir}/sample${i}"
#   for GNN in 400 401 403 405 419 426 427 429 430 431 432 433
#   do
#     cd  "${dir}/sample${i}/optimize_grid_b_140_phi_0.03-GNN${GNN}"
#     if [ -d 'equilibration' ]
#     then
#       pwd
#       tar -czf equilibration.tar.gz equilibration
#       rm -r equilibration
#     fi
#     if [ -d 'production_out' ]
#     then
#       pwd
#       tar -czf production_out.tar.gz production_out
#       rm -r production_out
#     fi
#     rm smatrix.txt
#     rm experimental_hic.npy
#   done
# done

# dir='/home/erschultz/downsampling_analysis'
# cd $dir
# for i in {208..224}
# do
#   for exp in 4
#   do
#     for j in {0..29}
#     do
#       cd "$dir/samples_exp${exp}/sample${i}/optimize_grid_b_180_v_8_spheroid_1.5-max_ent10"
#       cd "iteration${j}"
#       pwd
#       if [ -d "equilibration" ]
#       then
#         tar -czf equilibration.tar.gz equilibration
#         rm -r equilibration
#       fi
#       rm S.npy
#       rm L.npy
#       rm experimental_hic.npy
#       rm matrix*.png
#       rm D.npy
#       if [ -d "production_out" ]
#       then
#         tar -czf production_out.tar.gz production_out
#         rm -r production_out
#       fi
#     done
#   done
# done


dir='/home/erschultz/dataset_02_04_23/samples'
cd $dir
for i in {201..282}
do
  for k in 5 10
  do
    for j in {0..29}
    do
      cd  "${dir}/sample${i}"
      cd "optimize_grid_b_180_v_8_spheroid_1.5-max_ent${k}"
      cd "iteration${j}"
      pwd
      if [ -d "equilibration" ]
      then
        tar -czf equilibration.tar.gz equilibration
        rm -r equilibration
      fi
      rm S.npy
      rm L.npy
      rm experimental_hic.npy
      rm matrix*.png
      rm D.npy
      if [ -d "production_out" ]
      then
        tar -czf production_out.tar.gz production_out
        rm -r production_out
      fi
    done
  done
done

# dir='/home/erschultz/dataset_06_29_23/samples'
# cd $dir
# for i in 1 2 3 4 5 101 102 103 104 105 601 602 603 604 605
# do
#   cd  "${dir}/sample${i}"
#   pwd
#   rm -r optimize_grid_b_180_phi_0.008_spheroid_1.5-max_ent5
# done

# for dataset in "dataset_04_28_23" "dataset_08_17_23" "dataset_08_25_23" "dataset_09_10_23" "dataset_09_11_23" "dataset_09_16_23" "dataset_09_17_23" "dataset_09_18_23" "dataset_09_19_23" "dataset_09_25_23" "dataset_09_26_23"
# do
#   dir="/home/erschultz/${dataset}/samples"
#   cd $dir
#   for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 1 41 15 16 17 18 19 20 1753 1936 2834 3464 981 324
#   do
#     cd  "${dir}/sample${i}"
#     pwd
#     rm -r optimize_grid_b_140*
#     rm -r optimize_grid_b_261*
#     rm -r *longer
#   done
# done
#
# dir='/home/erschultz/dataset_02_04_23/samples'
# cd $dir
# for i in {201..282}
# do
#  cd  "${dir}/sample${i}"
#  rm -r optimize_grid_b_180_phi_0.01*
# done
