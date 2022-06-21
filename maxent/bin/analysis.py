import epilib as ep
import matplotlib.pyplot as plt
import numpy as np

sim = ep.Sim("production_out")

ep.plot_oe(ep.get_oe(sim.hic))
plt.savefig("oe.png")

sim.plot_tri()
plt.savefig("tri.png")

sim.plot_tri(vmaxp=np.mean(sim.hic)/2)
plt.savefig("tri_dark.png")

sim.plot_diff()
plt.savefig("diff.png")

sim.plot_scatter()
plt.savefig("scatter.png")

sim.plot_energy()
plt.savefig("energy.png")

plt.figure()
sim.plot_obs(diag=False)
plt.savefig("obs.png")

plt.figure()
plt.plot(sim.config['diag_chis'], 'o')
plt.savefig("diag_chis.png")
