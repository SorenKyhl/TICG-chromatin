from pathlib import Path

from pylib import default, epilib, parameters, hic, utils
from pylib.optimize_grid_size import optimize_stiffness, optimize_grid_size
from pylib.ideal_chain import ideal_chain_simulation
from pylib.maxent import Maxent
from pylib.config import Config
from pylib.pysim import Pysim

def tune_stiffness(nbeads_large, nbeads_small, pool_fn):
    """
    simualte ideal chain at large scale,
    pool large ideal hic down to small scale,
    tune stiffness of ideal chain at small scale so that
    diagonal probability of small simulation matches that of the
    pooled large simulation. 
    """
    factor = int(nbeads_large/nbeads_small)
    data_out = "data_out"

    try:
        ideal_chain_large = ideal_chain_simulation(nbeads_large)
        ideal_chain_large.run(data_out)
        sim_analysis = epilib.Sim(ideal_chain_large.root/data_out)
    except FileExistsError:
        large_out = "ideal-chain-"+str(nbeads_large) + "/" + data_out
        sim_analysis = epilib.Sim(large_out)

    try:
        ideal_chain_small = ideal_chain_simulation(nbeads_small)
        ideal_chain_small.run(data_out)
        small_config = ideal_chain_small.config
    except FileExistsError:
        small_config_file = "ideal-chain-"+str(nbeads_small) + "/config.json"
        small_config = utils.load_json(small_config_file)

    hic_pooled = pool_fn(sim_analysis.hic, factor)

    k_angle_opt =  optimize_stiffness(small_config, hic_pooled, low_bound=0, high_bound=1)
    return k_angle_opt

def scaleup(nbeads_large, nbeads_small, pool_fn):
    """
    run maximum entropy at small scale, 
    and use parameters to run production run at large scale.
    requires tuning the grid size and stiffness at small scale 
    in order for the chi parameters to be transferrable
    """
    config = parameters.get_config(nbeads_small)
    pipe = default.data_pipeline.resize(nbeads_small)
    gthic = pipe.load_hic(default.HCT116_hic)

    # tune grid size
    try:
        optimal_grid_size = optimize_grid_size(config, gthic)
    except:
        tmp_config = utils.load_json("optimize-grid-size/iteration5/config.json")
        optimal_grid_size = tmp_config['grid_size']

    grid_bond_ratio = optimal_grid_size/config['bond_length'] # for later, when getting large sim config

    # tune grid size
    try:
        k_angle_opt = tune_stiffness(nbeads_large, nbeads_small, pool_fn)
    except FileExistsError:
        config_opt = utils.load_json("optimize-stiffness/iteration5/config.json")
        k_angle_opt = config_opt['k_angle']
    
    # maxent at small size
    config['k_angle'] = k_angle_opt
    seqs = epilib.get_sequences(gthic, 10)
    goals = epilib.get_goals(gthic, seqs, config)
    params = default.params
    params["goals"] = goals

    me_root = "me-"+str(nbeads_small)
    try:
        me = Maxent(me_root,params, config, seqs, gthic)
        me.fit()
    except FileExistsError:
        pass
    
    # production at large simulation size
    final_it = utils.get_last_iteration(me_root)
    config_opt = Config(final_it/"config.json")

    config_large = parameters.get_config(nbeads_large, config_opt.config)

    gthic_large = pipe.resize(nbeads_large).load_hic(default.HCT116_hic)
    seqs_large = epilib.get_sequences(gthic_large, 10)

    sim_large_root = f"final-{nbeads_large}"
    sim_large = Pysim(sim_large_root, config_large, seqs_large, gthic=gthic_large)
    sim_large.run_eq(10000, 50000, 7)

if __name__ == "__main__":
    scaleup(2048, 1024, hic.pool_sum)

