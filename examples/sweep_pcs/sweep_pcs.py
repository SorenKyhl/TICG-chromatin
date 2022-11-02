import numpy as np
import json
import functools

from pylib.pipeline import Pipeline
from pylib import epilib as ep

"""
optimizes chromosome structure for HCT116-chr2-0Mbp-102.4Mbp using 1024 beads.
uses principal component analysis to assign bead types.
sweeps through the number of PCs, from 2 to 10
"""

config = json.load(open("config.json"))
config['bond_length'] = 30

gthic = np.load("experimental_hic.npy")

params = json.load(open("params.json"))
params["equilib_sweeps"] = 10000
params["production_sweeps"] = 50000
params["parallel"] = 7
params["trust_region"] = 1000
params["num_iterations"] = 12

for i in range(2,10):
    seqs_method = functools.partial(ep.get_sequences, k=i)
    name = "quick-" + str(i)
    pipe = Pipeline(name, gthic, config, params, seqs_method=seqs_method, load_first=False)
    pipe.fit()
