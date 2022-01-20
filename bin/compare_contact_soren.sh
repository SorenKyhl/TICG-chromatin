y='' # reference contact map file path
yhat='' # simulated contact map file path
dir='' # directory to write data to

python3 ~/TICG-chromatin/scripts/compare_contact_soren.py --y $y --yhat $yhat --dir $dir

# SCC should write to '${dir}/diastance_pearson.json'
