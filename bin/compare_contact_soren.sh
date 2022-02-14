idir='/home/eric/dataset_test/samples'

y="${idir}/sample70/y.npy" # reference contact map file path
yhat="${idir}/sample74/y.npy" # simulated contact map file path
odir="${idir}/sample74" # directory to write data to

python3 ~/TICG-chromatin/scripts/compare_contact_soren.py --y $y --yhat $yhat --dir $odir

# SCC should write to "${dir}/diastance_pearson.json"
