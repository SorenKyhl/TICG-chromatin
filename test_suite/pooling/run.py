
import numpy as np
from pylib.ideal_chain import ideal_chain_simulation
from pylib import epilib, hic

nbeads = 1024
sim = ideal_chain_simulation(nbeads)
sim.config["nSweeps"] = 10000

no_pooling_out = "no_pooling"
sim.config["conservative_contact_pooling"] = True
sim.run(no_pooling_out)

pooling_out = "pooling"
sim.config["contact_resolution"] = 2
sim.config["conservative_contact_pooling"] = True
sim.run(pooling_out)

pooling_out_nonconservative = "pooling_nonconservative"
sim.config["contact_resolution"] = 2
sim.config["conservative_contact_pooling"] = False
sim.run(pooling_out_nonconservative)

sim_pool = epilib.Sim("ideal-chain-1024/pooling/")
sim_nopool = epilib.Sim("ideal-chain-1024/no_pooling/")
sim_pool_nonconservative = epilib.Sim("ideal-chain-1024/pooling_nonconservative")

pooled_conservative = epilib.get_diagonal(hic.pool(sim_nopool.hic, 2))
assert(np.allclose(sim_pool.d, pooled_conservative))

pooled_nonconservative = epilib.get_diagonal(hic.pool_sum(sim_nopool.hic, 2))
assert(np.allclose(sim_pool_nonconservative.d, pooled_nonconservative))
