import numpy as np
import os.path as osp
from multiprocessing import Process
from scipy import optimize

from pylib import epilib, utils, default
from pylib.pysim import Pysim

""" 
module for optimizing grid size.

Context:
    The diagonal interactions determined by maximum entropy optimization are sensitive
    to the size of the grid used in TICG simulations. This is especially true for diagonal
    interaction parameters governing beads separated by short contour lengths (small s). 

    Often it is useful to modify the grid size in such a way that the prbability of 
    contact between nearest neighbors along the chain (i.e. p(s=1)) for a chain with no
    nonbonded interactions is identical to the probability observed in experiment. In 
    this case, the maximum entropy method will need to make only minor 
    adjustments to the diagonal interactions at short contour lengths. 

    In the converse situation, large attractive or repulsive interactions are needed to recapitulate
    the experimental p(s) curve for small s, and sometimes the experimental probabilities are
    not achievable, because nearest neighbor beads cannot be brought into more (or less)
    frequent contact simply becuase the grid spacing is too large or too small relative 
    to the bond length. In this case, the maximum entropy method does not converge.

    To ameliroate these issues, the grid size can be optimized prior to maximum entropy,
    to ensure stable behavior of the maximum entropy optimization.
"""


def nearest_neighbor_contact_error(grid_bond_ratio, sim_engine, gthic):
    """ calculate the difference between simulated and experimental nearest-neighbor contact
    probability. nearest neighbor contact probability is equal to p(s) evalulated at s=1,
    for s defined as the contour length in units of monomer units: s $\in$ [0,nbeads]
    
    args: 
        grid_bond_ratio [float]: grid size expressed as a ratio of the bond length.
        sim_engine [Pysim]: object to call simulation runs
        gthic [ndarray]: ground truth hic
    returns:
        error [float]: difference between simulated and experimental contact frequency
                        for nearest-neighbor beads: p_simulated(1) - p_experiment(1)
    """
    try:
        nearest_neighbor_contact_error.counter += 1
    except AttributeError:
        nearest_neighbor_contact_error.counter = 1

    output = f"iteration{nearest_neighbor_contact_error.counter}"
    print(f"optimizing grid size, {output}")
    sim_engine.config['grid_size'] = sim_engine.config['bond_length']*grid_bond_ratio

    process = Process(target=sim_engine.run, args=[output])
    process.start()
    process.join()
    #sim_engine.run(output)
    output_dir = osp.join(sim_engine.root,output)
    sim_analysis = epilib.Sim(output_dir, maxent_analysis=False)
    p1_sim = sim_analysis.d[1]
    p1_exp = epilib.get_diagonal(gthic)[1]
    error = p1_sim - p1_exp
    return error
     

def optimize_grid_size(config, gthic):
    """ tune grid size until simulated nearest neighbor contact probability
    is equal to the same probability derived from the ground truth hic matrix
    nearest neighbor contact probability is equal to p(s) evalulated at s=1,
    for s defined as the contour length in units of monomer units: s $\in$ [0,nbeads]

    args:
        config [json]: simulation config file. grid_size will be overwritten,
                            so it's initial value doesn't matter
        gthic [ndarray]: ground truth hic map. used as the target for optimization
    returns:
        optimal_grid_size [float]: size of grid cell that matches first neighbor contact probability
    """

    config['load_configuration'] = False
    config['nonbonded_on'] = False

    seqs = np.zeros((config['nspecies'], config['nbeads'])) # seqs don't matter, just set to zero.
    sim_engine = Pysim(root, config, seqs)

    low_bound = 0.75
    high_bound = 1.25
    result = optimize.brentq(nearest_neighbor_contact_error, low_bound, high_bound, args=(sim_engine, gthic), xtol=1e-2)
    optimal_grid_size = result * config['bond_length'] # optimizer returns the grid_to_bond ratio... have to convert to real units here.
    return optimal_grid_size


if __name__ == "__main__":
    """directory set up:
    config.json
    experimental_hic.npy [optional - if doesn't exist, will load from default pipeline. 
    """
    root = "optimize-grid-size"
    config = utils.load_json("config.json")
    config['nSweeps'] = 10000

    gthic_path = "experimental_hic.npy"
    if osp.exists(gthic_path):
        gthic = np.load(gthic_path)
    else:
        pipe = default.data_pipeline.resize(config['nbeads'])
        gthic = pipe.load_hic(default.HCT116_hic)

    optimal_grid_size = optimize_grid_size(config, gthic)
    print("optimal grid size is:", optimal_grid_size)





