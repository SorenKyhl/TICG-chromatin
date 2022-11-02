import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8,6]
plt.rcParams.update({'font.size':18})


import pylib.maxent as me
from pylib import epilib as ep
import pylib.utils

class Pipeline:
    
    def __init__(self, name, gthic, config, params, load_first=False, seqs_method=ep.get_sequences, goals_method=ep.get_goals):    
        self.name = name
        self.root = Path.cwd()/self.name
        self.gthic = gthic
        self.config = config
        self.params = params
        self.seqs_method = seqs_method
        self.goals_method = goals_method
        self.load_first = load_first  

    def get_seqs(self):
        if self.load_first:
            self.seqs = self.load_sequences()
        else:
            self.seqs = self.seqs_method(self.gthic)

    def get_goals(self):
        if self.load_first:
            self.goals = self.load_goals()
        else: 
            self.goals = self.goals_method(self.gthic, self.seqs, beadvol=self.config['beadvol'], grid_size=self.config['grid_size'])
    
    def fit(self):            
        self.get_seqs()
        self.get_goals()
        self.params["goals"] = self.goals
        
        optimizer = me.Maxent(root=self.root,
                        params=self.params, 
                        config=self.config, 
                        seqs=self.seqs, 
                        gthic=self.gthic)

        optimizer.run()
        
    def load_sequences(self):
        seqs = []
        for file in self.config["bead_type_files"]:
            print("loading", file)
            seqs.append(np.loadtxt(file) )
        seqs = np.array(seqs)
        return seqs

    def load_goals(self):
        goals_plaid  = np.loadtxt("obj_goal.txt")
        goals_diag = np.loadtxt("obj_goal_diag.txt")
        return np.hstack((goals_plaid, goals_diag))
    
