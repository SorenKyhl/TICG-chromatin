import numpy as np
import matplotlib.pyplot as plt

from pylib.pysim import Pysim
from pylib import utils
from pylib import epilib

config = utils.load_json("config.json")
config['nSweeps'] = 100000 # modify desired config settings

# calculate sequences from ground truth hic
gthic = np.load("experimental_hic.npy") 
seqs = epilib.get_sequences(gthic, k=5)

#... or load them if they're already written to disk
"""
dir_containing_seqs = "."
with utils.cd(dir_containing_seqs):
    utils.load_sequences(config)
"""

# construct simulation object
sim = Pysim(root=".", config=config, seqs=seqs, mkdir=False)

# run single simulation
sim.run("output_dir") # if output_dir is none, defaults to "data_out"

# run equilibration, followed by production 
"""
sim.run_eq(eq_sweeps=10000, prod_sweeps=50000, parallel=1) 
# if parallel = n (>1), it will run n production simulations in "core{n}"
# and aggregate the results in "production_out"
"""

# optional analysis
sim_out = epilib.Sim("output_dir")
sim_out.plot_contactmap()
plt.savefig("simulated_contactmap.png")
