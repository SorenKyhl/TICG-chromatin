import json
import logging
import os
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path

import jsbeautifier
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sympy import solve, symbols

"""
utility functions
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
    with open(path, "w") as f:
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        opts = jsbeautifier.default_options()
        opts.indent_size = 2
        f.write(jsbeautifier.beautify(json.dumps(data), opts))
        # json.dump(data, f, indent=4)


def cat(outfilename, infilenames, header=False):
    """implementation of linux cat command, concatenates ``infilenames`` into ``outfilename``

    Args:
        outfilename (str): destination for concatenated contents of ``infilenames``
        infilenames (List[str]): name of files to concatenate into ``outfilename``
    """
    with open(outfilename, "w") as outfile:
        first = True
        for infilename in infilenames:
            with open(infilename) as infile:
                if not first and header:
                    line = infile.readline() # skip header line for subsequent files
                for line in infile:
                    if line.strip():
                        # ignore lines that are purely whitespace
                        outfile.write(line)
            first = False


import subprocess


def copy_last_snapshot(xyz_in, xyz_out, nbeads):
    """copies final snapshot from xyz_in to xyz_out"""
    fout = open(xyz_out, "w")
    nlines = nbeads + 2
    subprocess.run(["tail", f"-{nlines}", xyz_in], stdout=fout)


def process_parallel(tasks, args):
    """process multiple tasks, each with the arguments (tuple)"""
    running_tasks = [Process(target=task, args=args) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()


def process_parallel_xargs(tasks, args):
    """process multiple tasks, each with different arguments (list of tuples)"""
    running_tasks = [
        Process(target=task, args=(arg,)) for task, arg in zip(tasks, args)
    ]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()


@contextmanager
def cd(newdir):
    """implementation of linux cd command using context manager"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def load_sequences(config):
    """load sequences from files specified in config file"""
    sequences = []
    for file in config["bead_type_files"]:
        logging.info("loading", file)
        sequences.append(np.loadtxt(file))
    sequences = np.array(sequences)
    return sequences


def write_sequences(sequences, config):
    assert len(sequences) == len(config["bead_type_files"])
    for seq, file in zip(sequences, config["bead_type_files"]):
        np.savetxt(file, seq)


def load_sequences_from_dir(dirname):
    dirname = Path(dirname)
    config = load_json(dirname / "config.json")
    with cd(dirname):
        sequences = load_sequences(config)
    return sequences


def uncorrelate_seqs(seqs):
    """transform sequences so that they are uncorrelated using cholesky transformation.
    following this blog post:
        https://blogs.sas.com/content/iml/2012/02/08/use-the-cholesky-transformation-to-correlate-and-uncorrelate-variables.html
    """
    sigma = np.cov(seqs)
    L = np.linalg.cholesky(sigma)
    seqs_uncorrelated = np.linalg.solve(L, seqs)
    return seqs_uncorrelated


def load_chis(config):
    try:
        nspecies = config["nspecies"]
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        chi = np.zeros((nspecies, nspecies))
        for i in range(nspecies):
            for j in range(nspecies):
                if j >= i:
                    chi[i, j] = config["chi" + letters[i] + letters[j]]
                    chi[j, i] = config["chi" + letters[i] + letters[j]]
    except KeyError:
        indices = np.triu_indices(config["nspecies"])
        chi = np.array(config["chis"])[indices]

    return chi

def plot_image(x, dark=False):
    x = np.array(x)
    v = x.flatten()
    lim = np.max([np.abs(np.min(v)), np.max(v)])
    if dark:
        lim /= 2
    plt.imshow(x, vmin=-lim, vmax=lim, cmap="bwr")
    plt.colorbar()

def nan_pearsonr(x, y):
    na_ind = np.logical_or(np.isnan(x), np.isnan(y))
    return pearsonr(x[~na_ind], y[~na_ind])

def pearson_round(x, y, stat = 'pearson', round = 2):
    "Wrapper function that combines np.round and pearsonr."
    if stat == 'pearson':
        fn = pearsonr
    elif stat == 'nan_pearson':
        fn = nan_pearsonr
    elif stat == 'spearman':
        fn = spearmanr
    x = np.array(x)
    y = np.array(y)
    assert x.shape == y.shape, f'shape mismatch, {x.shape} != {y.shape}'
    stat, _ = fn(x, y)
    return np.round(stat, round)

def make_composite(lower, upper):
    m, _ = lower.shape
    assert m == upper.shape[0]
    indu = np.triu_indices(m)
    indl = np.tril_indices(m)

    # make composite contact map
    composite = np.zeros((m, m))
    composite[indu] = upper[indu]
    composite[indl] = lower[indl]
    np.fill_diagonal(composite, 1)

    return composite

def triu_to_full(arr, m = None):
    '''Convert array of upper triangle to symmetric matrix.'''
    # infer m given length of upper triangle
    if m is None:
        l, = arr.shape
        x, y = symbols('x y')
        y=x*(x+1)/2-l
        result=solve(y)
        m = int(np.max(result))

    # need to reconstruct from upper traingle
    y = np.zeros((m, m))
    y[np.triu_indices(m)] = arr
    y += np.triu(y, 1).T

    return y

def newton(lam, obj_goal, B, gamma, current_chis, trust_region, method, norm=False):
    """newton's method"""
    obj_goal = np.array(obj_goal)
    lam = np.array(lam)

    if norm:
        obj_goal /= obj_goal
        lam /= obj_goal
        B /= np.outer(obj_goal, obj_goal)

    difference = obj_goal - lam  # pyright: ignore
    Binv = np.linalg.pinv(B)
    if method == "n":
        step = Binv @ difference
        if norm:
            step /= obj_goal
    elif method == "g":
        step = difference
    elif method == "n_new":
        step = newton_trust_region(difference, B, trust_region, log=True)
        step *= gamma
        howfar = np.sqrt(difference @ difference) / np.sqrt(obj_goal @ obj_goal)
        new_chis = current_chis + step
        return new_chis, howfar
    else:
        raise ValueError("specify method: n (newton), g (gradient descent), n_new (new newton)")

    steplength = np.sqrt(step @ step)

    logging.debug("========= step before gamma: ", steplength)
    logging.debug("obj goal", obj_goal)
    logging.debug("lam: ", lam)
    logging.debug("difference: ", difference)
    logging.debug("step: ", step)
    logging.debug("B: ", B)

    step *= gamma
    steplength = np.sqrt(step @ step)

    logging.debug("========= step after gamma: ", steplength)
    logging.debug("step: ", step)

    if trust_region is not None and steplength > trust_region:
        step /= steplength
        step *= trust_region
        steplength = np.sqrt(step @ step)

        logging.debug("======= OUTSIDE TRUST REGION =========")
        logging.debug("========= steplength: ", steplength)
        logging.debug("========= trust_region: ", trust_region)
        logging.debug("step: ", step)
        logging.debug("lam: ", lam)

    new_chis = current_chis - step
    # logging.debug(f"new chi values: {new_chis}\n")

    howfar = np.sqrt(difference @ difference) / np.sqrt(obj_goal @ obj_goal)

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


def clean_diag_chis(config):
    """set beginning diagonal chis to zero"""
    diag_chis = np.array(config["diag_chis"])
    diag_chis = np.clip(diag_chis, 0, 1e6)

    for i, chi in enumerate(diag_chis):
        if chi > 0:
            diag_chis[i] = 0
        else:
            break

    config["diag_chis"] = diag_chis.tolist()
    return config

def newton_trust_region(gradient, hessian, trust_region, log=False):
    """returns optimal step for trust region newton's method subproblem

    if the full step is within the trust region, take it
    otherwise, find the optimal point on the trust region boundary

    See chapter 4,
    Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer Science & Business Media, 2006.
    """
    full_step = -np.linalg.inv(hessian)@gradient
    if log:
        logging.info(f"full_step size: {np.linalg.norm(full_step)}")
    if np.linalg.norm(full_step) < trust_region:
        if log:
            logging.info("taking full step")
        return full_step
    else:
        """newton's method "subproblem" described in Nocedal"""

        if log:
            logging.info("taking trust region step")

        eigenvalues, eigenvectors = np.linalg.eig(hessian)
        lowest_eigenvalue = min(eigenvalues) # for some reason evals are not sorted!

        logging.info(f"lowest eval: {lowest_eigenvalue}")
        logging.info(eigenvalues)

        # initial lambda needs to be slightly more than lowest eigenvalue
        lamda = -0.9*lowest_eigenvalue

        for i in range(10):
            L = np.linalg.cholesky(hessian + lamda*np.eye(len(hessian)))
            R = L.T
            p = np.linalg.solve(R.T@R, -gradient)
            q = np.linalg.solve(R.T, p)

            if log:
                logging.info(f"----- stepsize: {np.sqrt(p@p)}, trust region: {trust_region}, lambda: {lamda}")

            lamda = lamda + (p@p)/(q@q) * (np.linalg.norm(p) - trust_region)/trust_region
        return p
