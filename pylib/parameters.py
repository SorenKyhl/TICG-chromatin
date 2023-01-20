import numpy as np
import copy

def get_physical_parameters(config, nbeads=None, base="gaussian"):
    """
    calculates physical parameters for a simulation with nbeads beads,
    and assigns them to config.
    if nbeads is not specified, take from config

    base = ["gaussian", "persistent"]
        if gaussian, chain is lp = 16.5, b = 16.5
        if persistent, chain is lp = 50, b = 16.5
    """
    if nbeads is None:
        nbeads = config["nbeads"]

    config = copy.deepcopy(config)
    

    assert(base in ["gaussian", "gaussian-5k", "persistent", "persistent-5k"])
    if base == "gaussian":
        baseb = 16.5
        baseg= 0.95*baseb
        baseN = 512000
        basev = 520
    elif base == "persistent":
        baseb = 100
        baseg= 0.95*baseb
        baseN = 84480
        basev = 3151.52
    elif base == "gaussian-5k":
        baseb = 82.5
        baseg= 0.95*baseb
        baseN = 20480
        basev = 13000
    elif base == "persistent-5k":
        baseb = 203.10
        baseg= 0.95*baseb
        baseN = 20480
        basev = 13000
    
    factor = (nbeads/baseN)**(-1/3)
    config["nbeads"] = nbeads
    config["bond_length"] = baseb * factor
    config["grid_size"] = baseg * factor
    config["beadvol"] = basev * baseN/nbeads
    
    return config

def interpolate(delta):

    data = np.loadtxt("data/dssWLCparams.txt")
    print(data)

if __name__ == "__main__":
    interpolate(3)
