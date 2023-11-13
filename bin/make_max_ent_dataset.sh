#! /bin/bash

odir="/home/erschultz/dataset_02_04_23_max_ent"
mkdir $odir
odir="/home/erschultz/dataset_02_04_23_max_ent/samples"
mkdir $odir


dir="/home/erschultz/dataset_02_04_23/samples"
for i in {201..282}
do
  odir_i="${odir}/sample${i}"
  mkdir $odir_i

  dir_i="${dir}/sample${i}/optimize_grid_b_180_phi_0.008_spheroid_1.5-max_ent5_700k/iteration30"
  cd $dir_i
  cp 'y.npy' "${odir_i}/y.npy"
  cp "${dir}/sample${i}/y.png" "${odir_i}/y.png"
  cp "${dir}/sample${i}/import.log" "${odir_i}/import.log"
done
