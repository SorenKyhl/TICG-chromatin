from pathlib import Path
import shutil
import json
import numpy as np
import hicstraw
import matplotlib.pyplot as plt
import pandas as pd
import jsbeautifier
import os
from multiprocessing import Process 
import time
from typing import Union

from pylib.pyticg import Sim
from pylib import analysis
from pylib.utils import cd, cat, copy_last_snapshot, newton
from pylib import utils
from pylib import epilib as ep

"""
pysim
"""
from contextlib import redirect_stdout
import subprocess

class Pysim:
    
    def __init__(self, 
            root : str, 
            config : dict, 
            seqs : list[list],
            gthic = None,
            randomize_seed : bool = True, 
            mkdir : bool = True, 
            setup_needed : bool = True):
        
        self.set_root(root, mkdir)
        self.set_config(config)
        self.seqs = seqs
        self.setup_needed = setup_needed # should be true unless instantiated using from_directory

        if gthic is not None:
            np.save(self.root/"experimental_hic.npy", gthic)

        if randomize_seed:
            self.randomize_seed()            

    @classmethod
    def from_directory(cls, sim_dir : str, new_root : str = None):
        """ construct a simulation object from a directory that's already set up (i.e. contains config and sequence files)

        dir: simulation directory from which to initialize
        root: new simulation root directory
        """
        sim_dir = Path(sim_dir).absolute()
        config = utils.load_json(sim_dir/"config.json")
        with utils.cd(sim_dir):
            seqs = utils.load_sequences(config)

        if new_root == None:
            return cls(sim_dir, config, seqs, mkdir=False, setup_needed=False)
        else:
            return cls(new_dir, config, seqs, mkdir=True, setup_needed=True)


    def set_root(self, root : str, mkdir : bool):
        """ set the root of the simulation. and (optionally) create a directory at that path

        args:
            root: simulation root, where all simulation outputs will be stored
            mkdir: if true, will make a new directory at the root path
        """

        self.root = Path(root)
        if mkdir:
            self.root.mkdir(exist_ok=False)


    def set_config(self, config : Union[dict, str]):
        """ set the config, either by assignment or by loading from a json file"""
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = utils.load_json(path)  

            
    def randomize_seed(self):
        """ sets seed for random number generator used in Monte Carlo simulation """
        self.config["seed"] = np.random.randint(1e5)
        

    def setup(self):
        """ write simulation inputs in simulation root directory, but only if setup_needed flag is on """
        # needs to happen regardless of setup_needed setting, because seed is randomized
        utils.write_json(self.config, Path(self.root, "config.json")) 

        # further setup needed - write seuences to directory, where they are read by simulation engine
        if self.setup_needed:
            if self.seqs is None:
                return 

            if self.seqs.ndim > 1:
                for i, seq in enumerate(self.seqs):
                    self.write_sequence(seq, Path(self.root, f"pcf{i+1}.txt"))
            else:
                self.write_sequence(self.seqs, Path(self.root, f"pcf1.txt"))

            
    def flatten_chis(self):
        """ restructure chi parameters into a 1-dimensional list
        returns: [1 x n] list = [plaid_chis, diag_chis]
        """
        indices = np.triu_indices(self.config["nspecies"])
        plaid_chis = np.array(self.config["chis"])[indices]
        diag_chis = np.array(self.config["diag_chis"])
        flat_chis = np.hstack((plaid_chis, diag_chis))
        return flat_chis
    
    def chis_to_matrix(self, flat_chis):
        """ restructure 1-D list of plaid chis into matrix 
        returns: [n x n] chi matrix
        """
        nspecies = self.config["nspecies"]
        X = np.zeros((nspecies, nspecies))
        X[np.triu_indices(nspecies)] = flat_chis
        X = X + X.T - np.diag(np.diag(X))
        return X 
    
    def split_chis(self, allchis):
        """ splits 1-D list of chis into: plaid_chis, diag_chis """ 
        nplaidchis = len(allchis) - len(self.config["diag_chis"])
        plaid_chis_flat, diag_chis = np.split(allchis, [nplaidchis])
        return plaid_chis_flat, diag_chis
    
    def set_chis(self, allchis):
        """ takes 1d vector of all chis and updates config chi parameters """
        plaid_chis_flat, diag_chis = self.split_chis(allchis)
        self.config["chis"] = self.chis_to_matrix(plaid_chis_flat).tolist()
        self.config["diag_chis"] = diag_chis.tolist() 
            
    def load_observables(self, jacobian=False):
        """ load observable trajectories from simulation output.
        return mean of observables throughout the simulation, 
        and (optionally) the jacobian of the observable matrix
        """
        obs_files = []
        obs_files.append(self.root/self.data_out/"observables.traj")
        obs_files.append(self.root/self.data_out/"diag_observables.traj")
        
        df_total = pd.DataFrame()
        for file in obs_files:
            df = pd.read_csv(file, delimiter="\t", header=None)
            df = df.dropna(axis=1)
            df = df.drop(df.columns[0] ,axis=1) 
            df_total = pd.concat((df_total, df), axis=1)
         
        self.obs = df_total.mean().values
        
        if jacobian:
            self.jac = df_total.cov().values
            return self.obs, self.jac
        else:
            return self.obs
    
    def run(self, name=None):
        """ run simulation """  
        self.data_out = name
        self.setup()
        with cd(self.root):
            if name:  
                engine = Sim(self.data_out) # output to dir: name
            else:
                engine = Sim()  # output to dir: data_out
                self.data_out = "data_out" # don't set earlier...
                # in this case, use default constructor so engine.run() pipes to stdout
            engine.run()
            
    def run_eq(self, equilibrium_sweeps : int, production_sweeps : int, parallel_simulations : int = 1):
        """ run equilibration followed by production simulation
        the production run can be executed in parallel, by specifying the parallel_simulations argument

        args:
            equilibrium_sweeps: number of equilibrium simulation sweeps
            production_sweeps: number of production simulation sweeps (per core)
            parallel_simulations: number of parallel production simulations to execute
        """
        equilibration_dir = "equilibration"
        production_dir = "production_out"

        # equilibration
        self.config["nSweeps"] = equilibrium_sweeps
        self.config["load_configuration"] = False
        self.run(equilibration_dir)
        
        # production. copy the last structure from equilibration to initialize production simulations
        equilibrated_structure = "equilibrated.xyz"
        copy_last_snapshot(xyz_in = (self.root/equilibration_dir/"output.xyz"), 
                           xyz_out = (self.root/equilibrated_structure), 
                           nbeads = self.config['nbeads'])
        
        self.config["nSweeps"] = production_sweeps
        self.config["load_configuration"] = True
        self.config["load_configuration_filename"] = equilibrated_structure
        
        if parallel_simulations == 1:
            self.run(production_dir)
        else:
            self.run_parallel(production_dir, parallel_simulations)
            
    def run_parallel(self, name : str, cores : int):
        """ run production run, using several parallel simulations with different initial seeds
        each parallel core dumps output to their own directory.
        after completion, the individual simulation data are aggregated into a final production directory 

        args:
            name: name of output file for aggregated simulation data
            cores: number of parallel simulations to execute
        """
        self.data_out = name
        print("reading from:", self.data_out)
        processes = []
        # create processes
        for i in range(cores):
            target = self.run
            args = (f"core{i}",)
            processes.append(Process(target=target, args=args))                 

        # run simulations
        for p in processes:  
            self.randomize_seed()
            p.start()     # run simulation for process p
            sleeptime = int(self.config['nbeads']/1000) # larger simulations take longer to move over
            time.sleep(sleeptime) # engine needs time to load config.json before next iteration
        
        # wait to join
        for p in processes:
            p.join()
            
        self.aggregate_production_files()
        
        
    def aggregate_production_files(self):
        """ aggregate simulation data from each core into final production folder.
        when simulations are run in parallel, each core dumps its output to a separate folder
        after all simulations are finished, 
        - concatenate all observables and energies into one trajectory file
        - sum all contact maps to produce one aggregate contact map
        """
        Path(self.root/"production_out").mkdir()
        
        # aggregate files are concatenated together to form one trajectory
        aggregate_files = ["observables.traj", "diag_observables.traj",
                           "energy.traj", "output.xyz"]
        for file in aggregate_files:
            cat(self.root/"production_out"/file, self.root.glob("core*/"+str(file)))
            
        # contact maps are summed together to form one contact map
        contact_files = list(self.root.glob("core*/contacts.txt"))
        self.combine_contactmaps(contact_files, output_file=self.root/"production_out/contacts.txt")
        
        production = Pysim(self.root/"production_out", self.config, self.seqs, mkdir=False)
        production.setup() # save config, seqs to production_out #TODO is this necessary?
        
        #TODO: add delete old files option
           
    def combine_contactmaps(self, contact_files: list, output_file : str = None):
        """ combines (sums) multiple contact maps from separate parallel simulations into one contactmap

        args:
            contact_files: list of files containing contact maps to be combined
            output_file (optional): name for output combined contact map. if specified, return type is None
        returns:
            aggregate contact map (if output_file is None) otherwise, saves contact map to file and returns None
        """
        combined = np.loadtxt(contact_files[0])
        for file in contact_files[1:]:
            combined += np.array(pd.read_csv(file, header=None, sep=" ")) # much faster than np.loadtxt
            #combined += np.loadtxt(file)

        if output_file:
            np.savetxt(output_file, combined, fmt="%d", delimiter=" ")  
        else:
            return combined  
    
    def write_sequence(self, sequence: list, path: str):
        """ write sequence of polymer bead types to disk at specified path """
        np.savetxt(path, sequence, fmt="%.8f", delimiter=" ")
    
    
