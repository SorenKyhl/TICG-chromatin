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

from pylib import analysis
from pylib.utils import cd, newton
from pylib.pysim import Pysim
from pylib import utils
from pylib import epilib as ep

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8,6]
plt.rcParams.update({'font.size':18})

"""
Maxent
"""
class Maxent:
    def __init__(self, root, params, config, seqs, gthic, overwrite=False, lengthen_iterations=False, analysis_on=True):
        """
        root: root of maxent filesystem
        params: maxent parameters
        config: default simulation config
        seqs: simulation sequences
        gthic: ground truth hic
        overwrite: will overwrite existing files
        """
        
        self.root = Path(root)
        self.resources = Path(self.root, "resources")
        self.params = params
        self.config = config
        self.seqs = seqs
        self.gthic = gthic
        self.overwrite = overwrite
        
        self.update_defualt_config()
        self.defaultsim = Pysim(self.resources, self.config, self.seqs, mkdir=False)
        
        self.chis = self.defaultsim.flatten_chis()
        self.loss = np.array([])
        self.track_plaid_chis, self.track_diag_chis = self.defaultsim.split_chis(self.chis)

        self.dampen_first_step = True
        self.lengthen_iterations = lengthen_iterations
        self.analysis_on = analysis_on
        
        
    def update_defualt_config(self):
        """
        ensure that the default config has the correct size chi matrix, nspecies,
        and number of bead_type_files
        """
        self.k, self.n  = np.shape(self.seqs)
        
        self.config['nbeads'] = self.n
        self.config['diag_cutoff'] = self.n
        #if (np.shape(self.config['chis']) != (self.k, self.k)):
        self.config['chis'] = np.zeros((self.k, self.k))
                                             
        self.config['nspecies'] = self.k
        
        bead_type_files = []
        for i in range(self.k):
            bead_type_files.append(f"pcf{i+1}.txt")
        self.config['bead_type_files'] = bead_type_files
        
    
    def make_directory(self):
        """ create maxent directory and populate resources """
        self.root.mkdir(exist_ok=self.overwrite)
        self.resources.mkdir(exist_ok=self.overwrite)
        np.save(self.resources/"experimental_hic.npy", self.gthic)
        utils.write_json(self.config, self.resources/"config.json")
        # TODO: write seqs, defaultsim.config, goals, gthic to resources. 
        # or: maybe just pickle the maxent instance? 
        
    def update_state(self, newchis, newloss, sim):
        self.chis = np.vstack((self.chis, newchis))
        self.loss = np.append(self.loss, newloss)
        
        plaid, diag = sim.split_chis(newchis)
        self.track_plaid_chis = np.vstack((self.track_plaid_chis, plaid))
        self.track_diag_chis = np.vstack((self.track_diag_chis, diag))

        np.savetxt(self.root/"chis.txt", plaid, fmt="%.4f", newline=" ")
        np.savetxt(self.root/"chis_diag.txt", diag, fmt="%.4f", newline= " ")
        np.savetxt(self.root/"convergence.txt", self.loss, fmt="%.16f")
        np.save(self.root/"plaid_chis.npy", self.track_plaid_chis)
        np.save(self.root/"daig_chis.npy", self.track_diag_chis)
        
        plt.figure()
        plt.plot(self.loss,  '.-')
        plt.savefig(self.root/"loss.png")
        
        plt.figure()
        plt.plot(self.track_plaid_chis,  '.-')
        plt.savefig(self.root/"track_plaid_chis.png")
        
        plt.figure()
        plt.plot(self.track_diag_chis,  '.-')
        plt.savefig(self.root/"track_diag_chis.png") 
        
        
    def analyze(self):
        if self.analysis_on:
            analysis.main()
        
    def run(self):
        """ execute maxent optimization """ 
        
        self.make_directory()
        newchis = self.defaultsim.flatten_chis() # initial chis
        utils.write_json(self.params, self.root/"params.json")
        
        for it in range(self.params["num_iterations"]):
            
            sim = Pysim(root = self.root/f"iteration{it}", 
                        config = self.defaultsim.config,
                        seqs = self.defaultsim.seqs,
                        randomize_seed = True)
            
            sim.set_chis(newchis)
  
            if self.lengthen_iterations and (it>0):
                self.params["production_sweeps"] = int(1.1*self.params["production_sweeps"]) 
                
            sim.run_eq(self.params["equilib_sweeps"], 
                       self.params["production_sweeps"], 
                       self.params["parallel"])
        
            curr_chis = sim.flatten_chis()
            obs, jac = sim.load_observables(jacobian=True)
            
            if self.dampen_first_step:
                # dampen the first step so it doesn't overshoot
                if (it == 0):
                    gamma = 0.25*self.params["gamma"]
                else:
                    gamma = self.params["gamma"]
                
            newchis, newloss = newton(lam = obs, 
                                      obj_goal = self.params["goals"], 
                                      B = jac, 
                                      gamma = gamma, 
                                      current_chis = curr_chis, 
                                      trust_region = self.params["trust_region"], 
                                      method = "n")
            
            self.update_state(newchis, newloss, sim)
            os.symlink(self.resources/"experimental_hic.npy", sim.root/"experimental_hic.npy")
            with cd(sim.root):
                self.analyze()
                
            print(f"{it}: {newloss}")
        
    
    def set_config(self, path):
        """ load config from path """
        self.config = self.load_json(path)
                     
    def load_json(self, path):
        with open(path) as f:
            myjson = json.load(f)
        return myjson
    
    def write_json(self, data, path):
        with open(path, 'w') as f:
            opts = jsbeautifier.default_options()
            opts.indent_size = 2
            f.write(jsbeautifier.beautify(json.dumps(data), opts))
            json.dump(data, f, indent=4)
