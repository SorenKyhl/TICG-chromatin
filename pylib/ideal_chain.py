import sys

from pylib import parameters
from pylib.pysim import Pysim

""" ideal_chain
module for running ideal chain simulations
"""

def ideal_chain_simulation(nbeads, grid_bond_ratio=None):
    if grid_bond_ratio is None:
        config = parameters.get_config(nbeads)
    else:
        config = parameters.get_config(nbeads, grid_bond_ratio=grid_bond_ratio)

    config['nonbonded_on'] = False
    config['load_bead_types'] = False
    sim = Pysim(root=f"ideal-chain-{nbeads}", config=config, seqs=None)
    return sim

if __name__ == "__main__":
    if len(sys.argv)>1:
        nbeads = int(sys.argv[1])
    else: 
        print("nbeads not specified,")
        print("usage: bonded_only_simulation nbeads")
    sim = ideal_chain_simulation(nbeads)
    sim.run()

