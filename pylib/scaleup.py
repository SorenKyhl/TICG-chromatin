import logging

import matplotlib.pyplot as plt
import numpy as np

from pylib import parameters
from pylib.config import Config
from pylib.ideal_chain import ideal_chain_simulation
from pylib.Maxent import Maxent
from pylib.optimize import optimize_config, optimize_stiffness
from pylib.Pysim import Pysim
from pylib.utils import default, epilib, hic_utils, utils


def plot_stiffness_error(ideal_small, ideal_large, gthic_big):
    """
    ideal_small: small sim with stiffness
    ideal_large: big sim without stiffness
    gthic_big: ground truth hic at big scale
    """
    factor = int(len(gthic_big) / len(ideal_small.hic))
    gthic_small = hic_utils.pool_sum(gthic_big, factor)

    factor = int(len(ideal_large.hic) / len(ideal_small.hic))
    id_pooled = hic_utils.pool_sum(ideal_large.hic, factor)

    ratio_pooled = epilib.get_diagonal(id_pooled) / epilib.get_diagonal(gthic_small)
    ratio_sim = ideal_small.d / epilib.get_diagonal(gthic_small)

    ratio2 = ratio_sim / ratio_pooled

    error = np.mean(ratio2[100:700])

    # optional:
    # epilib.plot_diagonal(gthic2k, 'k', scale='loglog')
    # epilib.plot_diagonal(id2k.d, label="ideal 2k")

    # epilib.plot_diagonal(gthic1k, 'k')
    epilib.plot_diagonal(ideal_small.d, label="stiff optimal", scale="loglog")
    epilib.plot_diagonal(id_pooled, label="pooled ideal 2k")
    plt.legend()
    plt.title("p(s) stiff vs pooled")
    plt.savefig("stiff_p.png")

    plt.figure()
    epilib.plot_diagonal(ratio_sim, label="stiff", scale="loglog")
    epilib.plot_diagonal(ratio_pooled, label="pooled")
    plt.legend()
    plt.title("p(s) ratio stiff vs pooled")
    plt.savefig("stiff_ratios.png")

    plt.figure()
    epilib.plot_diagonal(ratio2, scale="loglog")
    epilib.plot_diagonal(np.ones(1024), "k")
    plt.title(f"ratio of ratios, mean: {error}")
    plt.savefig("stiff_ratio_ratios.png")


def tune_stiffness(nbeads_large, nbeads_small, pool_fn, grid_bond_ratio, method, large_contact_pooling_factor, match_ideal_large_grid=False):
    """optimize chain stiffness of small system to match p(s) curve of large system

    simualte ideal chain at large scale and pool large ideal hic down to small scale,
    tune stiffness of ideal chain at small scale so that
    diagonal probability of small simulation matches that of the pooled large simulation.

    match_ideal_large_grid: tune small simulation to match first point of pooled ideal large chain
        (this is not the same as the first point of pooled large gthic)
    """
    data_out = "data_out"

    # run large ideal simulation
    try:
        ideal_chain_large = ideal_chain_simulation(nbeads_large, grid_bond_ratio)
        if nbeads_large >= 10240:
            ideal_chain_large.config["nSweeps"] = 25000
        ideal_chain_large.config["contact_resolution"] = large_contact_pooling_factor
        ideal_chain_large.run(data_out)
        large_hic = np.loadtxt(ideal_chain_large.root / data_out / "contacts.txt")
    except FileExistsError:
        large_out = "ideal-chain-" + str(nbeads_large) + "/" + data_out
        large_hic = np.loadtxt(large_out + "/contacts.txt")

    # get config for small ideal chain simulation
    ideal_chain_small = ideal_chain_simulation(nbeads_small, grid_bond_ratio)
    small_config = ideal_chain_small.config

    factor = int(len(large_hic)/nbeads_small)
    # don't use nbeads_large, because the large_hic might be pooled during simulation
    large_ideal_hic_pooled  = pool_fn(large_hic, factor)

    if match_ideal_large_grid:
        # tune small simulation to match first point of pooled ideal large chain
        # (this is not the same as the first point of pooled large gthic)

        # tune grid size
        root = "optimize-grid-match-ideal-large"
        try:
            optimal_grid_size = optimize_config(small_config, large_ideal_hic_pooled, 'grid', 0.5, 2.5, root)
        except FileExistsError:
            optimal_grid_size = utils.load_json(f"{root}/config.json")[
                "grid_size"
            ]

        small_config["grid_size"] = optimal_grid_size

    k_angle_opt = optimize_stiffness(
        small_config, large_ideal_hic_pooled, low_bound=0, high_bound=2.5, method=method
    )

    if match_ideal_large_grid:
        return k_angle_opt, optimal_grid_size
    else:
        return k_angle_opt



def scaleup(nbeads_large, nbeads_small, pool_fn, method="notbayes", pool_large = True,
        zerodiag = False, match_ideal_large_grid=False):


if __name__ == "__main__":
    scaleup(2048, 1024, hic_utils.pool)
