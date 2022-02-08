idir='/home/eric/sequences_to_contact_maps/dataset_09_21_21/samples/sample2'

y="${idir}/y.npy" # reference contact map file path
yhat="${idir}/yhat-5.npy" # simulated contact map file path
odir="${idir}/5" # directory to write data to

python3 ~/TICG-chromatin/scripts/compare_contact_soren.py --y $y --yhat $yhat --dir $odir

# SCC should write to "${dir}/diastance_pearson.json"
