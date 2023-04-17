import copy
import functools
import json
import os
import os.path as osp

import numpy as np
from pylib import epilib
from pylib.pysim import Pysim
from scipy import optimize
from sklearn.metrics import mean_squared_error


def nearest_neighbor_contact_error(grid_bond_ratio, sim_engine, gthic):
    """calculate the difference between simulated and experimental nearest-neighbor contact
    probability.

    Args:
        grid_bond_ratio [float]: grid size expressed as a ratio of the bond length.
        sim_engine [Pysim]: object to call simulation runs
        gthic [ndarray]: ground truth hic

    Returns:
        error [float]: difference between simulated and experimental contact frequency
                        for nearest-neighbor beads: p_simulated(1) - p_experiment(1)
    """
    try:
        nearest_neighbor_contact_error.counter += 1
    except AttributeError:
        nearest_neighbor_contact_error.counter = 1

    output = f"iteration{nearest_neighbor_contact_error.counter}"
    print(f"optimizing grid size, {output}")
    sim_engine.config["grid_size"] = sim_engine.config["bond_length"] * grid_bond_ratio

    sim_engine.run(output)

    output_dir = osp.join(sim_engine.root, output)
    sim_analysis = epilib.Sim(output_dir, maxent_analysis=False)
    p1_sim = sim_analysis.d[1]
    p1_exp = epilib.get_diagonal(gthic)[1]
    error = p1_sim - p1_exp
    print(f"error, {error}")
    return error


def optimize_grid_size(config, gthic, low_bound=0.5, high_bound=2, root="optimize-grid-size"):
    """tune grid size until simulated nearest neighbor contact probability
    is equal to the same probability derived from the ground truth hic matrix.

    Args:
        config [json]: simulation config file. grid_size will be overwritten,
                            so it's initial value doesn't matter
        gthic [ndarray]: ground truth hic map. used as the target for optimization
    Returns:
        optimal_grid_size [float]: size of grid cell that matches first neighbor contact probability
    """
    config = copy.deepcopy(config)

    config["load_configuration"] = False
    config["nonbonded_on"] = False
    config["load_bead_types"] = False
    config['angles_on'] = False
    config['k_angle'] = 0
    config['double_count_main_diagonal'] = False
    config['conservative_contact_pooling'] = False


    gthic /= np.mean(np.diagonal(gthic))
    sim_engine = Pysim(root, config, seqs=None, overwrite=False)

    result = optimize.brentq(
        nearest_neighbor_contact_error,
        low_bound,
        high_bound,
        args=(sim_engine, gthic),
        xtol=1e-3,
        maxiter=10,
    )
    # optimizer returns the grid_to_bond ratio... have to convert to real units.
    optimal_grid_size = result * config["bond_length"]
    return optimal_grid_size

def main():
    dir = '/home/erschultz'
    os.chdir(osp.join(dir, 'Su2020/samples/sample1003'))
    with open("config.json") as f:
        config = json.load(f)
    config["nSweeps"] = 20000

    gthic = np.load("y.npy")

    optimal_grid_size = optimize_grid_size(config, gthic)
    print("optimal grid size is:", optimal_grid_size)


if __name__ == "__main__":
    main()
