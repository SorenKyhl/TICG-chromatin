import numpy as np   
import json
from contextlib import contextmanager
import os
import jsbeautifier

""" 
utils 
"""

def load_json(path):
    with open(path) as f:
        myjson = json.load(f)
    return myjson

def write_json(data, path):
    """
    warning: this mutates the original json... 
    converts any numpy arrays into lists so that the parser can write them out.
    """
    with open(path, 'w') as f:
        
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        
        opts = jsbeautifier.default_options()
        opts.indent_size = 2
        f.write(jsbeautifier.beautify(json.dumps(data), opts))
        #json.dump(data, f, indent=4)

def cat(outfilename, infilenames):
    with open(outfilename, 'w') as outfile:
        for infilename in infilenames:
            with open(infilename) as infile:
                for line in infile:
                    if line.strip():
                        outfile.write(line)    

import subprocess
def copy_last_snapshot(xyz_in, xyz_out, nbeads):
    """copies final snapshot from xyz_in to xyz_out"""
    fout = open(xyz_out, "w")
    nlines = nbeads+2
    subprocess.run(["tail", f"-{nlines}", xyz_in], stdout=fout)


def process_parallel(tasks, args):
    """process multiple tasks, each with the arguments (tuple)"""
    running_tasks = [Process(target=task, args=args) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()
        
def process_parallel_xargs(self, tasks, args):
    """process multiple tasks, each with different arguments (list of tuples)"""
    running_tasks = [Process(target=task, args=(arg,)) for task, arg in zip(tasks, args)]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()   

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        
    
def load_sequences(config):
    seqs = []
    for file in config["bead_type_files"]:
        print("loading", file)
        seqs.append(np.loadtxt(file) )
    seqs = np.array(seqs)
    return seqs

def load_chis(config):
    try:
        nspecies = config['nspecies']
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        chi = np.zeros((nspecies, nspecies))
        for i in range(nspecies):
            for j in range(nspecies):
                if j >= i:
                    chi[i,j] = config["chi" + letters[i] + letters[j]]
                    chi[j,i] = config["chi" + letters[i] + letters[j]]
    except KeyError:
        indices = np.triu_indices(config["nspecies"])
        chi = np.array(config["chis"])[indices]

    return chi


""" 
newton's method
"""

def newton(lam, obj_goal, B, gamma, current_chis, trust_region, method, multiplicity=1):
    obj_goal = np.array(obj_goal)
    lam = np.array(lam)
    
    difference = obj_goal - lam
    Binv = np.linalg.pinv(B)
    if method == "n":
        step = Binv@difference
    elif method == "g":
        step = difference

    steplength = np.sqrt(step@step)
    
    print("========= step before gamma: ", steplength)
    print('obj goal', obj_goal)
    print('lam: ', lam)
    print('difference: ', difference)
    print('step: ', step)
    print('B: ', B)

    step *= gamma
    steplength = np.sqrt(step@step)
    
    print("========= step after gamma: ", steplength)
    print('step: ', step)
    
    if steplength > trust_region:
        step /= steplength
        step *= trust_region
        steplength = np.sqrt(step@step)     
        
        print("======= OUTSIDE TRUST REGION =========")
        print("========= steplength: ", steplength)
        print("========= trust_region: ", trust_region)
        print('step: ', step)
        print('lam: ', lam)
        
    new_chis = current_chis - multiplicity*step
    # print(f"new chi values: {new_chis}\n")

    howfar = np.sqrt(difference@difference)/np.sqrt(obj_goal@obj_goal)

    return new_chis, howfar
