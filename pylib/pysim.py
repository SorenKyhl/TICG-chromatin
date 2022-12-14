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
    
    def __init__(self, root, config, seqs, randomize_seed=False, mkdir=True, setup_needed=True):
        self.set_root(root, mkdir)
        self.set_config(config)
        self.seqs = seqs

        self.setup_needed = setup_needed # should be true unless instantiated using from_directory

        if randomize_seed:
            self.randomize_seed()            

    @classmethod
    def from_directory(cls, root):
        """constructor that can initialize from directory that's already set up"""
        #root = Path.cwd()/root
        root = Path(root).absolute()
        config = utils.load_json(root/"config.json")
        with utils.cd(root):
            seqs = utils.load_sequences(config)
        return cls(root, config, seqs, mkdir=False, setup_needed=False)

    def set_root(self, root, mkdir):
        self.root = Path(root)
        if mkdir:
            self.root.mkdir(exist_ok=False)

    def set_config(self, config):
        """ load config from path 
        config: [dict, filepath]
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = utils.load_json(path)  
            
    def randomize_seed(self):
        """sets seed for random number generator used in Monte Carlo simulation"""
        self.config["seed"] = np.random.randint(1e5)
        
    def setup(self):
        """save simulation inputs in simulation root directory"""
        if self.setup_needed:
            utils.write_json(self.config, Path(self.root, "config.json"))       
            for i, seq in enumerate(self.seqs):
                self.write_seq(seq, Path(self.root, f"pcf{i+1}.txt"))
            
    def flatten_chis(self):
        """returns 1-D list: [plaid_chis, diag_chis]"""
        indices = np.triu_indices(self.config["nspecies"])
        plaid_chis = np.array(self.config["chis"])[indices]
        diag_chis = np.array(self.config["diag_chis"])
        flat_chis = np.hstack((plaid_chis, diag_chis))
        return flat_chis
    
    def chis_to_matrix(self, flat_chis):
        """converts flattened plaid chis to matrix"""
        nspecies = self.config["nspecies"]
        X = np.zeros((nspecies, nspecies))
        X[np.triu_indices(nspecies)] = flat_chis
        X = X + X.T - np.diag(np.diag(X))
        return X 
    
    def split_chis(self, allchis):
        """splits all chis into: [plaid_chis, diag_chis] """ 
        nplaidchis = len(allchis) - len(self.config["diag_chis"])
        plaid_chis_flat, diag_chis = np.split(allchis, [nplaidchis])
        return plaid_chis_flat, diag_chis
    
    def set_chis(self, allchis):
        """takes 1d vector of all chis and updates config chi parameters"""
        plaid_chis_flat, diag_chis = self.split_chis(allchis)
        self.config["chis"] = self.chis_to_matrix(plaid_chis_flat).tolist()
        self.config["diag_chis"] = diag_chis.tolist() 
            
    def load_observables(self, jacobian=False):
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
        """ run simulation"""  
        self.data_out = name
        self.setup()
        with cd(self.root):
            if name:  
                engine = Sim(self.data_out) # output to dir: name
            else:
                engine = Sim()  # output to dir: data_out
                self.data_out = "data_out" # don't set earlier...
                # want default constructor so engine.run() pipes to stdout
            engine.run()
            
    def run_eq(self, eq_sweeps, prod_sweeps, parallel=1):
        """run equilibration followed by production simulation"""
        eq_dir = "equilibration"
        prod_dir = "production_out"
        eq_snap = "equilibrated.xyz"
        
        # equilibration
        self.config["nSweeps"] = eq_sweeps
        self.config["load_configuration"] = False
        self.run(eq_dir)
        
        # production
        copy_last_snapshot(xyz_in = (self.root/eq_dir/"output.xyz"), 
                           xyz_out = (self.root/eq_snap), 
                           nbeads = self.config['nbeads'])
        
        self.config["nSweeps"] = prod_sweeps
        self.config["load_configuration"] = True
        self.config["load_configuration_filename"] = eq_snap
        
        if parallel == 1:
            self.run(prod_dir)
        else:
            print("running parallel")
            self.run_parallel(prod_dir, parallel)
            
    def run_parallel(self, name, cores):
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
        """aggregate simulation data from each core into final production folder"""
        Path(self.root/"production_out").mkdir()
        
        aggregate_files = ["observables.traj", "diag_observables.traj",
                           "energy.traj", "output.xyz"]
        for file in aggregate_files:
            cat(self.root/"production_out"/file, self.root.glob("core*/"+str(file)))
            
        contact_files = list(self.root.glob("core*/contacts.txt"))
        self.combine_contactmaps(contact_files, output_file=self.root/"production_out/contacts.txt")
        
        production = Pysim(self.root/"production_out", self.config, self.seqs, mkdir=False)
        production.setup() # save config, seqs to production_out #TODO is this necessary?
        
        #TODO: add delete old files option
           
    def combine_contactmaps(self, contact_files, output_file=None):
        """combines contact maps in contact_files into one contactmap"""
        combined = np.loadtxt(contact_files[0])
        for file in contact_files[1:]:
            combined += np.loadtxt(file)

        if output_file:
            np.savetxt(output_file, combined, fmt="%d", delimiter=" ")  
        else:
            return combined  
    
    def write_seq(self, seq, path):
        np.savetxt(path, seq, fmt="%.8f", delimiter=" ")
    
    
