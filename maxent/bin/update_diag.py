import json
import numpy as np
import sys

it = sys.argv[1]

config_file = "iteration"+it+"/config.json"

with open(config_file) as f:
    config = json.load(f)

#assert(config['diagonal_on'] == True)
allchis = np.loadtxt('chis_diag.txt')

# get last row of 'chis.txt'
lastchis = list(allchis[int(it)])

config['diag_chis'] = lastchis

with open(config_file, "w") as f:
    json.dump(config, f)


"""
if(config['plaid_on'] == True):
    nspecies = config['nspecies']
    ndiagchis = lastchis.size - nspecies*(nspecies+1)/2

    if(ndiagchis>0):
        config['diag_chis'] = lastchis[:-ndiagchis]

        with open("config.json", "w") as f:
            json.dump(config, f)
else:
    config['diag_chis'] = list(lastchis)

    with open("config.json", "w") as f:
        json.dump(config, f)
"""
