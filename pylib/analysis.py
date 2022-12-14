import numpy as np
import matplotlib.pyplot as plt


from pylib import epilib as ep
from pylib import utils, energy_utils
plt.rcParams['figure.figsize'] = [8,6]
plt.rcParams.update({'font.size':18})


def sim_analysis(sim):
    """analyze data from simulation only (doesn't require ground truth hic)"""
    error = sim.plot_consistency()
    plt.savefig("consistency.png")
    plt.close()
    if error > 0.01:
        print("SIMULATION IS NOT CONSISTENT")

    plt.figure()
    sim.plot_contactmap()
    plt.savefig("contactmap.png")
    plt.close()

    plt.figure()
    sim.plot_energy()
    plt.savefig("energy.png")
    plt.close()

    plt.figure()
    sim.plot_obs(diag=False)
    plt.savefig("obs.png")
    plt.close()

    plt.figure()
    plt.plot(sim.config['diag_chis'], 'o')
    plt.savefig("diag_chis.png")
    plt.close()

    plt.figure()
    utils.plot_image(sim.config['chis'])
    plt.savefig("chis.png")
    plt.close()

    plot_energy_matrices(sim)

    plt.figure()
    plot_chi_matrix(sim)
    plt.close()

    plt.figure()
    sim.plot_oe()
    plt.savefig("oe.png")
    plt.close()

    plt.figure()
    sim.plot_diagonal()
    plt.savefig("diagonal.png")
    plt.close()

    plt.figure()
    sim.plot_diagonal(scale="log")
    plt.savefig("diagonal-log.png")
    plt.close()

def compare_analysis(sim):
    """ analyze comparison of simulation with ground truth contact map"""
    plt.figure()
    ep.plot_tri(sim.hic, sim.gthic, oe=True)
    plt.savefig("tri_oe.png")
    plt.close()

    sim.plot_tri()
    plt.savefig("tri.png")
    plt.close()

    sim.plot_tri(log=True)
    plt.savefig("tri_log.png")
    plt.close()

    sim.plot_tri(vmaxp=np.mean(sim.hic)/2)
    plt.savefig("tri_dark.png")
    plt.close()

    sim.plot_diff()
    plt.savefig("diff.png")
    plt.close()

    sim.plot_scatter()
    plt.savefig("scatter.png")
    plt.close()

def maxent_analysis(sim):
    """analyze properties related to maxent optimization"""
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
    plt.close()
    
    plt.figure()
    RMSE_vec = np.loadtxt("../RMSE.txt")
    RMSLE_vec = np.loadtxt("../RMSLE.txt")
    plt.plot(RMSE_vec, label="RMSE")
    plt.plot(RMSLE_vec, label="RMSLE")
    plt.xlabel("iteration")
    plt.ylabel("RMSE")
    plt.savefig("../RMSE.png")
    plt.close()

    plt.figure()
    convergence = np.loadtxt("../convergence.txt")
    fig, axs = plt.subplots(3, figsize=(12,14))
    axs[0].plot(convergence)
    axs[0].set_title("Loss")
    axs[1].plot(RMSE_vec, label="RMSE")
    axs[1].plot(RMSLE_vec, label="RMSLE")
    axs[1].set_title("RMSE/RMSLE")
    axs[1].legend()
    axs[2].plot(SCC_vec)
    axs[2].set_title("SCC")
    plt.savefig("../error.png")
    plt.close()

    plt.figure()
    sim.plot_obs_vs_goal()
    plt.savefig("obs_vs_goal.png")

def plot_chi_matrix(sim):
    utils.plot_image(np.array(sim.config['chis']))

def plot_energy_matrices(sim):
    # energy matrices
    S, D, E, ED = energy_utils.calculate_all_energy(sim.config, sim.seqs.T, np.array(sim.config['chis']))

    plt.figure()
    utils.plot_image(S)
    plt.title("Smatrix")
    plt.savefig("matrix_S.png")
    plt.close()

    plt.figure()
    utils.plot_image(D)
    plt.title("Dmatrix")
    plt.savefig("matrix_D.png")
    plt.close()
    
    plt.figure()
    utils.plot_image(E)
    plt.title("Ematrix")
    plt.savefig("matrix_E.png")
    plt.close()

    plt.figure()
    utils.plot_image(ED)
    plt.title("EDmatrix")
    plt.savefig("matrix_ED.png")
    plt.close()

def main():
    sim = ep.Sim("production_out")
    sim_analysis(sim)
    compare_analysis(sim)
    maxent_analysis(sim)

if __name__ == "__main__":
    main()
