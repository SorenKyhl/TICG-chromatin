import numpy as np

from pylib.pysim import Pysim
from pylib.maxent import Maxent
from pylib import utils
from pylib import epilib as ep

config = utils.load_json("resources/config.json")
with utils.cd("resources"):
    sequences = utils.load_sequences(config)

config['dmatrix_on'] = True
config['ematrix_on'] = True
config['bond_length'] = 30

gthic = np.load("resources/experimental_hic.npy")

params = utils.load_json("resources/params.json")
params['equilib_sweeps'] = 10000
params['production_sweeps'] = 50000
params['trust_region'] = 12

goals = ep.get_goals(gthic, sequences, config['beadvol'], config['grid_size'])
#plaid = np.loadtxt("resources/obj_goal.txt")
#diag = np.loadtxt("resources/obj_goal_diag.txt")
params['goals'] = np.hstack((plaid,diag))

me = Maxent("fullscale-night", params, config, sequences, gthic, lengthen_iterations=False)
me.fit()
