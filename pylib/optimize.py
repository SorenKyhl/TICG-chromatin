import copy
import functools
import os
import os.path as osp

import numpy as np
from bayes_opt import BayesianOptimization
from pylib.Pysim import Pysim
from pylib.utils import default, hic_utils, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.epilib import Sim
from pylib.utils.xyz import (xyz_load, xyz_to_contact_distance,
                             xyz_to_contact_grid)
from scipy import optimize
from sklearn.metrics import mean_squared_error

"""
module for optimizing simulation parameters

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

    To ameliorate these issues, the grid size can be optimized prior to maximum entropy,
    to ensure stable behavior of the maximum entropy optimization.
"""

class ErrorMetric():
    """Calculate difference between sim and exp p(s) for different metrics."""
    def __init__(self, metric, mode, gthic, config, sim_engine, dataset=None):
        self.counter = 0 # number of iterations
        self.metric = metric
        self.mode = mode # what is being optimized
        self.gthic = gthic # experiment
        self.config = config
        self.sim_engine = sim_engine
        self.dataset = dataset
        self.xyz = get_bonded_simulation_xyz(config, self.dataset, False) # bonded xyz


    def __call__(self, val):
        '''
        Inputs:
            val [float]: current value of optimization

        Outut:
            error [float]: difference between simulated and experimental contact frequency
                            according to metric
        '''
        self.counter += 1
        output = f"iteration{self.counter}"
        print(f"optimizing {self.mode}, {output}")

        if self.mode.startswith("grid"):
            grid_size = self.config["bond_length"] * float(val)
            angle = self.config["k_angle"]
        elif self.mode.startswith("distance"):
            distance = self.config["bond_length"] * float(val)
        elif self.mode.startswith("angle"):
            angle = float(val)
            grid_size = self.config["grid_size"]
        else:
            raise Exception(f'Unrecognized mode {self.mode}')

        if self.xyz is not None and self.mode.startswith("grid"):
            y = xyz_to_contact_grid(self.xyz, grid_size, dtype=float)
            y /= np.mean(np.diagonal(y))
            self.mean_dist = DiagonalPreprocessing.genomic_distance_statistics(y)
        elif self.xyz is not None and self.mode.startswith("distance"):
            y = xyz_to_contact_distance(self.xyz, distance, dtype=float)
            y /= np.mean(np.diagonal(y))
            self.mean_dist = DiagonalPreprocessing.genomic_distance_statistics(y)
        else:
            self.sim_engine.config["grid_size"] = grid_size
            self.sim_engine.config["k_angle"] = angle

            # run simulation
            self.sim_engine.run(output)
            output_dir = osp.join(self.sim_engine.root, output)
            self.sim_analysis = Sim(output_dir, maxent_analysis=False)
            self.mean_dist = self.sim_analysis.d

        if self.metric.startswith('neighbor'):
            error = self.neighbor_error()
        elif self.metric == 'mse':
            error = self.mse_error()

        print(val, error)
        return error

    def neighbor_error(self):
        i = int(self.metric.split('_')[1])
        p_i_sim = self.mean_dist[i]
        p_i_exp = hic_utils.get_diagonal(self.gthic)[i]
        error = p_i_sim - p_i_exp
        return error

    def mse(self):
        p_s_sim = DiagonalPreprocessing.genomic_distance_statistics(self.sim_analysis.hic, 'prob')
        p_s_exp = DiagonalPreprocessing.genomic_distance_statistics(self.gthic, 'prob')
        error = mean_squared_error(p_s_sim, p_s_exp)
        return error

def get_bonded_simulation_xyz(config, dataset, throw_exception=False):
    if dataset is None:
        if throw_exception:
            raise Exception('dataset is None')
        else:
            print('get_bonded_simulation_xyz: dataset is None')
            return None

    boundary_type = config['boundary_type']
    if boundary_type == 'spheroid':
        boundary_type += f'_{config["aspect_ratio"]}'
    beadvol = config['beadvol']
    bond_type = config['bond_type']
    dir = osp.join(dataset, f'boundary_{boundary_type}/beadvol_{beadvol}/bond_type_{bond_type}')
    if bond_type == 'SC':
        k_bond = config['k_bond']
        dir = osp.join(dir, f'k_bond_{k_bond}')

    m = config['nbeads']
    b = config['bond_length']
    dir = osp.join(dir,f'm_{m}/bond_length_{b}')

    k_angle = config['k_angle']
    theta_0 = config['theta_0']
    if 'phi_chromatin' in config:
        phi = config['phi_chromatin']
        dir = osp.join(dir, f'phi_{phi}/angle_{k_angle}')
    elif 'target_volume' in config:
        v = config['target_volume']
        dir = osp.join(dir, f'v_{v}/angle_{k_angle}')
    else:
        raise Exception('One of phi_chromatin and target_volume is required')
    if theta_0 != 180:
        dir += f'_theta0_{theta_0}'

    if not osp.exists(dir):
        if throw_exception:
            raise Exception(f'{dir} does not exist')
        else:
            print(f'get_bonded_simulation_xyz: {dir} does not exist')
            return None

    xyz_file = osp.join(dir, 'production_out/output.xyz')
    xyz = xyz_load(xyz_file, multiple_timesteps=True)
    return xyz

def optimize_config(config, gthic, mode='grid', low_bound=0.1, high_bound=3,
                        root=None, metric='neighbor_1', dataset=None):
    """
    Tune grid size until simulated nearest neighbor contact probability
    is equal to the same probability derived from the ground truth hic matrix.

    Args:
        config [json]: simulation config file. grid_size will be overwritten,
                            so it's initial value doesn't matter
        gthic [ndarray]: ground truth hic map. used as the target for optimization
        mode [str]: optimization mode
        metric [str]: choice of error metric
    Returns:
        optimum [float]: optimal value of val corresponding to mode given metric
    """
    if root is None:
        root = f'optimize-{mode}'
    if not osp.exists(root):
        os.mkdir(root, mode=0o755)

    config = copy.deepcopy(config)
    config["load_configuration"] = False
    config["nonbonded_on"] = False
    config["load_bead_types"] = False
    gthic /= np.mean(np.diagonal(gthic))


    sim_engine = Pysim(root, config, seqs=None, overwrite=False, mkdir=False)
    error_metric = ErrorMetric(metric, mode, gthic, config, sim_engine, dataset)
    if metric.startswith('neighbor'):
        try:
            print(low_bound, high_bound)
            result = optimize.brentq(
                error_metric,
                low_bound,
                high_bound,
                xtol=1e-3,
                maxiter=15
            )
        except RuntimeError as e:
            print(e)
            return None
        except ValueError:
            low_bound /= 2
            high_bound *= 2
            result = optimize.brentq(
                error_metric,
                low_bound,
                high_bound,
                xtol=1e-3,
                maxiter=15,
            )
    else:
        try:
            result = optimize.minimize(
                error_metric,
                0.9,
                bounds=[(low_bound, high_bound)],
                args=(sim_engine),
                options={'maxiter':10},
            )
        except RuntimeError as e:
            print(e)
            raise
        result = result.x

    if mode in {'grid', 'distance'}:
        # optimizer returns the grid_to_bond ratio, have to convert to real units.
        optimum = result * config["bond_length"]
    elif mode == 'angle':
        optimum = result

    return optimum

def stiffness_root_error(hic, gthic):
    """calculate error for the purposes of optimizing stiffness.

    for lack of a better option, just try to match the p(s)
    at the  4th bead for N=1024
    """
    nbeads = len(hic)
    index = int(4*nbeads/1024) - 1
    return hic_utils.get_diagonal(hic)[index] - hic_utils.get_diagonal(gthic)[index]
    # return np.mean(hic_utils.get_diagonal(hic) - hic_utils.get_diagonal(gthic))

def stiffness_bayes_error(hic, gthic):
    """error used for bayesian optimization"""
    sim = hic_utils.get_diagonal(hic)
    exp = hic_utils.get_diagonal(gthic)
    return mean_squared_error(sim, exp, squared=False)

def simulate_stiffness_error(k_angle, sim_engine, gthic, method):
    """simulate and calculate the difference between simulated and experimental p(s)
    diagonal probability as a function of the chain stiffness defined by the angle potential.

    Args:
        k_angle [float]: grid size expressed as a ratio of the bond length.
        sim_engine [Pysim]: object to call simulation runs
        gthic [ndarray]: ground truth hic

    Returns:
        error [float]: difference between simulated and experimental p(s) diagonal probability
    """
    try:
        simulate_stiffness_error.counter += 1
    except AttributeError:
        simulate_stiffness_error.counter = 1

    output = f"iteration{simulate_stiffness_error.counter}"
    print(f"optimizing chain stiffness, {output}")
    sim_engine.config["k_angle"] = k_angle

    sim_engine.run(output)

    output_dir = osp.join(sim_engine.root, output)
    sim_analysis = Sim(output_dir, maxent_analysis=False)

    if method == "bayes":
        # bayes method maximizes, so return negative of error
        error = -stiffness_bayes_error(sim_analysis.hic, gthic)
        print(f"bayes error, {error}")
    else:
        error = stiffness_root_error(sim_analysis.hic, gthic)
        print(f"root error, {error}")
    return error

def optimize_stiffness(config, gthic, low_bound=0, high_bound=1, method="notbayes", root="optimize-stiffness"):
    """tune angle stiffness until simulated p(s) diagonal probabity
    is equal to the same probability derived from the ground truth hic matrix.

    Args:
        config [json]: simulation config file. grid_size will be overwritten,
                            so it's initial value doesn't matter
        gthic [ndarray]: ground truth hic map. used as the target for optimization
    Returns:
        optimal_k_angle[float]: angle stiffness
    """
    config = copy.deepcopy(config)
    config["load_configuration"] = False
    config["nonbonded_on"] = False
    config["load_bead_types"] = False

    sim_engine = Pysim(root, config, seqs=None, gthic=gthic)

    if method == "bayes":
        pbounds = {"k_angle": (low_bound, high_bound)}

        black_box = functools.partial(
            simulate_stiffness_error, sim_engine=sim_engine, gthic=gthic, method="bayes"
        )

        optimizer = BayesianOptimization(
            f=black_box,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )

        optimizer.maximize(
            init_points=2,
            n_iter=20,
        )

        return optimizer.max["params"]["k_angle"]  # pyright: ignore

    else:
        # use rootfinding
        method = "notbayes"
        result = optimize.brentq(
            simulate_stiffness_error,
            low_bound,
            high_bound,
            args=(sim_engine, gthic, method),
            xtol=1e-2,
            maxiter=10,
        )
        return result


if __name__ == "__main__":
    """directory set up:
    config.json
    experimental_hic.npy [optional - if doesn't exist, will load from default pipeline.
    """
    config = utils.load_json("config.json")
    config["nSweeps"] = 10000

    gthic_path = "experimental_hic.npy"
    if osp.exists(gthic_path):
        gthic = np.load(gthic_path)
    else:
        pipe = default.data_pipeline.resize(config["nbeads"])
        gthic = pipe.load_hic(default.HCT116_hic)

    optimal_grid_size = optimize_config(config, gthic)
    print("optimal grid size is:", optimal_grid_size)
