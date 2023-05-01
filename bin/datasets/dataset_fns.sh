#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh

param_setup

run() {
  # move utils to scratch
  scratchDirI="${scratchDir}/${i}"
  move

  check_dir

  argsFile="${dataFolder}/setup/sample_${i}.txt"

  # generate sequences
  python3 ~/TICG-chromatin/scripts/get_params.py --args_file $argsFile --method $seqMethod --exclusive $exclusive --m $m --p_switch $pSwitch --lmbda $lmbda --scale_resolution $scaleResolution --k $k --save_npy --seq_seed $seqSeed --chi=$chi --chi_method $chiMethod --chi_seed $chiSeed --min_chi $minChi --max_chi $maxChi --fill_diag $fillDiag --ensure_distinguishable --diag_chi_method $chiDiagMethod --diag_chi_slope $chiDiagSlope --diag_chi_scale $chiDiagScale --min_diag_chi $minDiagChi --max_diag_chi $maxDiagChi --diag_bins $diagBins --chi_constant=$chiConstant --chi_multiplier=$chiMultiplier --diag_chi_constant=$chiDiagConstant --diag_chi_midpoint=$chiDiagMidpoint --dense_diagonal_on $dense --dense_diagonal_cutoff $denseCutoff --dense_diagonal_loading $denseLoading --m_continuous $mContinuous --small_binsize $smallBinSize --big_binsize $bigBinSize --n_small_bins $nSmallBins --n_big_bins $nBigBins --diag_start $diagStart --diag_cutoff $diagCutoff > params.log

  # set up config.json
  python3 ~/TICG-chromatin/scripts/get_config.py --args_file $argsFile --phi_chromatin $phiChromatin --bond_length $bondLength --grid_size $gridSize --bead_vol $beadVol --k_angle $kAngle --m $m --n_sweeps $nSweeps --dump_stats_frequency $dumpStatsFrequency --dump_frequency $dumpFrequency --track_contactmap $trackContactMap --gridmove_on $gridMoveOn --update_contacts_distance $updateContactsDistance --TICG_seed $TICGSeed --constant_chi $constantChi --e_constant $eConstant --s_constant $sConstant --use_lmatrix $useL --use_smatrix $useS --use_dmatrix $useD --load_configuration_filename $initConfig --bond_type $bondType --parallel $parallel --num_threads $numThreads --dense_diagonal_on $dense > config.log

  # run simulation
  python3 ~/TICG-chromatin/bin/datasets/run.py > log.log

  # calculate contact map
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --save_npy --random_mode --plot > contact_map.log
  #
  # clean up
  mkdir -p $dir
  rm -f default_config.json *.xyz
  mv * $dir
  rm -d $scratchDirI
}
