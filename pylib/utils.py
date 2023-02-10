import numpy as np   
import json
from contextlib import contextmanager
import os
import jsbeautifier
from pathlib import Path
import matplotlib.pyplot as plt

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
    """python implementation of linux cat command, concatenates ``infilenames`` into ``outfilename``

    Args:
        outfilename (str): destination for concatenated contents of ``infilenames``
        infilenames (List[str]): name of files to concatenate into ``outfilename``
    """
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
    """load sequences from files specified in config file"""
    sequences  = []
    for file in config["bead_type_files"]:
        print("loading", file)
        sequences.append(np.loadtxt(file) )
    sequences  = np.array(sequences )
    return sequences 


def write_sequences(sequences, config):
    assert(len(sequences) == len(config["bead_type_files"]))
    for seq, file in zip(sequences, config["bead_type_files"]):
        np.savetxt(file, seq)


def load_sequences_from_dir(dirname):
    dirname = Path(dirname)
    config = load_json(dirname/"config.json")
    with cd(dirname):
        sequences = load_sequences(config)
    return sequences


def uncorrelate_seqs(seqs):
    """
    transform sequences so that they are uncorrelated using cholesky transformation.
    following this blog post:
        https://blogs.sas.com/content/iml/2012/02/08/use-the-cholesky-transformation-to-correlate-and-uncorrelate-variables.html
    """
    sigma = np.cov(seqs)
    L = np.linalg.cholesky(sigma)
    seqs_uncorrelated = np.linalg.solve(L, seqs)
    return seqs_uncorrelated


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


def plot_image(x):
    x = np.array(x)
    v = x.flatten()
    lim = np.max([np.abs(np.min(v)), np.max(v)])
    plt.imshow(x, vmin=-lim, vmax=lim, cmap='bwr')
    plt.colorbar()

""" 
newton's method
"""

def newton(lam, obj_goal, B, gamma, current_chis, trust_region, method):
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
        
    new_chis = current_chis - step
    # print(f"new chi values: {new_chis}\n")

    howfar = np.sqrt(difference@difference)/np.sqrt(obj_goal@obj_goal)

    return new_chis, howfar


def get_last_iteration(directory):
    """get path to final iteration of optimization directory

    Args:
        directory (str or path): directory containing iterations

    Returns:
        path to final iteration in ``directory``
    """
    iterations = Path(directory).glob("iteration*")
    iterations = list(iterations)
    iterations = sorted(iterations, key=lambda path: path.name[-1])
    return iterations[-1]
