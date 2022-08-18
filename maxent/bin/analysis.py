import epilib as ep
import numpy as np
import matplotlib.pyplot as plt

print("analysis")


'''
sim = ep.Sim("production_out")

SCC = ep.get_SCC(sim.hic, sim.gthic)
RMSE = ep.get_RMSE(sim.hic, sim.gthic)
RMSLE = ep.get_RMSLE(sim.hic, sim.gthic)

with open("../SCC.txt", "a") as f:
    f.write(str(SCC) + "\n")
    
with open("../RMSE.txt", "a") as f:
    f.write(str(RMSE) + "\n")

with open("../RMSLE.txt", "a") as f:
    f.write(str(RMSLE) + "\n")

plt.figure()
SCC_vec = np.loadtxt("../SCC.txt")
plt.plot(SCC_vec)
plt.xlabel("iteration")
plt.ylabel("SCC")
plt.savefig("../SCC.png")

plt.figure()
RMSE_vec = np.loadtxt("../RMSE.txt")
RMSLE_vec = np.loadtxt("../RMSLE.txt")
plt.plot(RMSE_vec, label="RMSE")
plt.plot(RMSLE_vec, label="RMSLE")
plt.xlabel("iteration")
plt.ylabel("RMSE")
plt.savefig("../RMSE.png")
'''

convergence = np.loadtxt("../convergence.txt")
fig, axs = plt.subplots(3, figsize=(12,14))
axs[0].plot(convergence)
axs[0].set_title("Loss")
#axs[1].plot(RMSE_vec, label="RMSE")
#axs[1].plot(RMSLE_vec, label="RMSLE")
axs[1].set_title("RMSE/RMSLE")
axs[1].legend()
#axs[2].plot(SCC_vec)
axs[2].set_title("SCC")
plt.savefig("../error.png")

'''
sim.plot_obs_vs_goal()
plt.savefig("obs_vs_goal.png")

error = sim.plot_consistency()
plt.savefig("consistency.png")
if error > 0.01:
    print("SIMULATION IS NOT CONSISTENT")

sim.plot_oe()
plt.savefig("oe.png")

plt.figure()
ep.plot_tri(sim.hic, sim.gthic, oe=True)
plt.savefig("tri_oe.png")

plt.figure()
sim.plot_diagonal()
plt.savefig("diagonal.png")

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
'''
