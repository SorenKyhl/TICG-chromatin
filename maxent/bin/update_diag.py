import json
import numpy as np
import sys
import argparse

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--it', type=int, help='current iteration')

    args = parser.parse_args()
    return args

def main():
    args = getArgs()

    config_file = "iteration{}/config.json".format(args.it)

    with open(config_file) as f:
        config = json.load(f)

    #assert(config['diagonal_on'])
    allchis = np.loadtxt('chis_diag.txt')

    # get last row of 'chis.txt'
    lastchis = list(allchis[int(args.it)])

    config['diag_chis'] = lastchis

    with open(config_file, "w") as f:
        json.dump(config, f)

if __name__ == '__main__':
    main()


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
