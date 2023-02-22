import numpy as np
import matplotlib.pyplot as plt

from pylib import default, epilib, parameters, hic, utils
from pylib.optimize import optimize_stiffness, optimize_grid_size
from pylib.ideal_chain import ideal_chain_simulation
from pylib.maxent import Maxent
from pylib.config import Config
from pylib.pysim import Pysim


def plot_stiffness_error(ideal_small, ideal_large, gthic_big):
    """
    ideal_small: small sim with stiffness
    ideal_large: big sim without stiffness
    gthic_big: ground truth hic at big scale
    """
    factor = int(len(ideal_large.hic) / len(ideal_small.hic))
    gthic_small = hic.pool_sum(gthic_big, factor)
    id_pooled = hic.pool_sum(ideal_large.hic, factor)

    ratio_pooled = epilib.get_diagonal(id_pooled) / epilib.get_diagonal(gthic_small)
    ratio_sim = ideal_small.d / epilib.get_diagonal(gthic_small)

    ratio2 = ratio_sim / ratio_pooled

    error = np.mean(ratio2[100:700])

    # optional:
    # epilib.plot_diagonal(gthic2k, 'k', scale='loglog')
    # epilib.plot_diagonal(id2k.d, label="ideal 2k")

    # epilib.plot_diagonal(gthic1k, 'k')
    epilib.plot_diagonal(ideal_small.d, label="stiff optimal")
    epilib.plot_diagonal(id_pooled, label="pooled ideal 2k")
    plt.legend()
    plt.title("p(s) stiff vs pooled")
    plt.savefig("stiff_p.png")

    plt.figure()
    epilib.plot_diagonal(ratio_sim, label="stiff")
    epilib.plot_diagonal(ratio_pooled, label="pooled")
    plt.legend()
    plt.title("p(s) ratio stiff vs pooled")
    plt.savefig("stiff_ratios.png")

    plt.figure()
    epilib.plot_diagonal(ratio2)
    epilib.plot_diagonal(np.ones(1024), "k")
    plt.title(f"ratio of ratios, mean: {error}")
    plt.savefig("stiff_ratio_ratios.png")


def tune_stiffness(nbeads_large, nbeads_small, pool_fn, grid_bond_ratio, method):
    """optimize chain stiffness of small system to match p(s) curve of large system

    simualte ideal chain at large scale and pool large ideal hic down to small scale,
    tune stiffness of ideal chain at small scale so that
    diagonal probability of small simulation matches that of the pooled large simulation.
    """
    factor = int(nbeads_large / nbeads_small)
    data_out = "data_out"

    try:
        ideal_chain_large = ideal_chain_simulation(nbeads_large, grid_bond_ratio)
        ideal_chain_large.run(data_out)
        large_hic = np.loadtxt(ideal_chain_large.root / data_out / "contacts.txt")
    except FileExistsError:
        large_out = "ideal-chain-" + str(nbeads_large) + "/" + data_out
        large_hic = np.loadtxt(large_out + "/contacts.txt")

    ideal_chain_small = ideal_chain_simulation(nbeads_small, grid_bond_ratio)
    small_config = ideal_chain_small.config

    hic_pooled = pool_fn(large_hic, factor)

    k_angle_opt = optimize_stiffness(
        small_config, hic_pooled, low_bound=0, high_bound=2, method=method
    )
    return k_angle_opt


def scaleup(nbeads_large, nbeads_small, pool_fn, method="bayes"):
    """optimize chis on small system, and scale up parameters to large system

    requires tuning the grid size and stiffness at small scale,
    in order for the chi parameters to be transferrable
    """
    config_small = parameters.get_config(nbeads_small)

    gthic_large = hic.load_hic(nbeads_large, pool_fn)
    gthic_small = hic.load_hic(nbeads_small, pool_fn)

    seqs_large = hic.load_seqs(nbeads_large, 10)
    seqs_small = hic.load_seqs(nbeads_small, 10)

    # tune grid size
    try:
        optimal_grid_size = optimize_grid_size(config_small, gthic_small)
    except FileExistsError:
        optimal_grid_size = utils.load_json("optimize-grid-size/config.json")[
            "grid_size"
        ]

    config_small["grid_size"] = optimal_grid_size
    grid_bond_ratio = (
        optimal_grid_size / config_small["bond_length"]
    )  # for later, when getting large sim config

    # tune grid size
    try:
        k_angle_opt = tune_stiffness(
            nbeads_large, nbeads_small, pool_fn, grid_bond_ratio, method
        )
    except FileExistsError:
        k_angle_opt = utils.load_json("optimize-stiffness/config.json")["k_angle"]

    config_small["k_angle"] = k_angle_opt

    # plot results
    final_it_stiff = utils.get_last_iteration("optimize-stiffness")
    ideal_small = epilib.Sim(final_it_stiff)
    ideal_large = epilib.Sim(f"ideal-chain-{str(nbeads_large)}/data_out")
    plot_stiffness_error(ideal_small, ideal_large, gthic_large)

    # maxent at small size
    goals = epilib.get_goals(gthic_small, seqs_small, config_small)
    params = default.params
    params["goals"] = goals

    me_root = "me-" + str(nbeads_small)
    try:
        me = Maxent(me_root, params, config_small, seqs_small, gthic_small)
        me.fit()
    except FileExistsError:
        pass

    # production at large simulation size
    final_it = utils.get_last_iteration(me_root)
    config_opt = Config(final_it / "config.json")

    config_large = parameters.get_config(
        nbeads_large, config_opt.config, grid_bond_ratio=grid_bond_ratio
    )
    config_large["k_angle"] = 0
    config_large["angles_on"] = False

    sim_large_root = f"final-{nbeads_large}"
    sim_large = Pysim(sim_large_root, config_large, seqs_large, gthic=gthic_large)
    sim_large.run_eq(10000, 50000, 7)


if __name__ == "__main__":
    scaleup(2048, 1024, hic.pool_sum)
