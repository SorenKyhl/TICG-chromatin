import copy
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pylib.analysis as analysis
import sympy
from pylib.Pysim import Pysim
from pylib.utils import utils

plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams.update({"font.size": 18})

PathLike = Union[str, Path]

"""
Maxent
TODO - move most recent iteration to "best" directory for easy access.
"""


class Maxent:
    def __init__(
        self,
        root: PathLike,
        params: dict,
        config: dict,
        seqs: np.ndarray,
        gthic: np.ndarray,
        overwrite: bool = False,
        lengthen_iterations: bool = False,
        analysis_on: bool = True,
        initial_chis: Optional[bool] = None,
        dampen_first_step: bool = True,
        final_it_sweeps: int = 0,
        mkdir: bool = True
    ):
        """
        root: root of maxent filesystem
        params: maxent parameters
        config: default simulation config
        seqs: simulation sequences
        gthic: ground truth hic
        overwrite: will overwrite existing files
        """

        # maxent things
        self.set_root(root)
        self.params = params
        self.config = config
        if seqs.shape[0] > seqs.shape[1]:
            # Maxent assumes seq is kxm
            self.seqs = seqs.T
        else:
            self.seqs = seqs
        self.gthic = gthic
        self.overwrite = overwrite
        self.mkdir = mkdir

        if "goals" not in self.params:
            raise ValueError("goals are not specified in parameters")

        self.update_default_config()
        self.defaultsim = Pysim(self.resources, self.config, self.seqs, mkdir=False)
        if initial_chis is None:
            self.initial_chis = self.defaultsim.flatten_chis()
        else:
            self.initial_chis = initial_chis
            self.defaultsim.set_chis(self.initial_chis)

        # tracking things
        self.chis = self.defaultsim.flatten_chis()
        self.loss = np.array([])
        self.track_plaid_chis, self.track_diag_chis = self.defaultsim.split_chis(
            self.chis
        )

        # optimization things
        self.dampen_first_step = dampen_first_step
        self.final_it_sweeps = final_it_sweeps
        self.lengthen_iterations = lengthen_iterations
        self.analysis_on = analysis_on

    def set_root(self, root: PathLike):
        """
        sets the root directory and other directories in the file tree
        """
        self.root = Path(root)
        self.resources = Path(self.root, "resources")

    def update_default_config(self):
        """
        ensure that the default config has the correct size chi matrix, nspecies,
        and number of bead_type_files
        """
        self.k, self.n = np.shape(self.seqs)

        self.config["nbeads"] = self.n
        self.config["diag_cutoff"] = self.n
        # if (np.shape(self.config['chis']) != (self.k, self.k)):
        self.config["chis"] = np.zeros((self.k, self.k))

        self.config["nspecies"] = self.k

        bead_type_files = []
        for i in range(self.k):
            bead_type_files.append(f"pcf{i+1}.txt")
        self.config["bead_type_files"] = bead_type_files

    def make_directory(self):
        """create maxent directory and populate resources"""
        if self.root.exists() and self.overwrite:
            shutil.rmtree(self.root)
        if self.mkdir:
            self.root.mkdir(exist_ok=False)
        self.resources.mkdir(exist_ok=False)
        np.save(self.resources / "experimental_hic.npy", self.gthic)
        utils.write_json(self.config, self.resources / "config.json")
        np.save(self.resources / "x.npy", self.seqs)
        # TODO: write seqs, defaultsim.config, goals, gthic to resources.
        # or: maybe just pickle the maxent instance?

    def track_progress(self, newchis, newloss, sim):
        """
        saves (and plots) parameter values and loss over the course of optimization
        """
        self.chis = np.vstack((self.chis, newchis))
        self.loss = np.append(self.loss, newloss)

        plaid, diag = sim.split_chis(newchis)
        self.track_plaid_chis = np.vstack((self.track_plaid_chis, plaid))
        self.track_diag_chis = np.vstack((self.track_diag_chis, diag))

        np.savetxt(self.root / "chis.txt", plaid, fmt="%.4f", newline=" ")
        np.savetxt(self.root / "chis_diag.txt", diag, fmt="%.4f", newline=" ")
        np.savetxt(self.root / "convergence.txt", self.loss, fmt="%.16f")
        np.save(self.root / "plaid_chis.npy", self.track_plaid_chis)
        np.save(self.root / "diag_chis.npy", self.track_diag_chis)

        converged = self.plot_convergence()
        self.plot_plaid_chis()
        self.plot_plaid_chis(True)
        self.plot_diag_chis()

        return converged

    def plot_convergence(self):
        plt.figure()
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        iterations = np.arange(1, len(self.loss)+1)
        plt.plot(iterations, self.loss, ".-")

        converged_it = None
        for i in range(1, len(self.loss)):
            diff = self.loss[i] - self.loss[i-1]
            if np.abs(diff) < self.eps and self.loss[i] < self.loss[0]:
                converged_it = iterations[i]
                break

        if converged_it is not None:
            plt.axvline(converged_it, color = 'k', label = 'converged')
            plt.legend()
            converged = True
        else:
            converged = False

        plt.tight_layout()
        plt.savefig(self.root / "loss.png")

        return converged

    def plot_plaid_chis(self, legend=False):
        k = sympy.Symbol('k')
        result = sympy.solvers.solve(k*(k-1)/2 + k - self.track_plaid_chis.shape[1])
        k = np.max(result) # discard negative solution

        counter = 0
        LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        plt.figure()
        for i in range(k):
            for j in range(k):
                if j < i:
                    continue
                chistr = f"chi{LETTERS[i]}{LETTERS[j]}"
                plt.plot(self.track_plaid_chis[:, counter], ".-", label = chistr)
                counter += 1
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel(r'$\chi_{IJ}$ value', fontsize=16)
        if legend:
            plt.legend(loc=(1.04,0), ncol = 3)
            plt.tight_layout()
            plt.savefig(self.root / "track_plaid_chis_legend.png")
        else:
            plt.savefig(self.root / "track_plaid_chis.png")
        plt.tight_layout()
        plt.close()

    def plot_diag_chis(self):
        plt.figure()
        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Diagonal Parameter Value", fontsize=16)
        plt.plot(self.track_diag_chis, ".-")
        plt.tight_layout()
        plt.savefig(self.root / "track_diag_chis.png")

    def analyze(self, dir):
        if self.analysis_on:
            analysis.main(dir)

    def fit(self):
        """execute maxent optimization"""
        t0 = time.time()
        self.make_directory()
        newchis = self.initial_chis
        utils.write_json(self.params, self.resources / "params.json")

        max_iterations = self.params["iterations"]
        self.eps = 1e-2
        if 'conv_defn' in self.params:
            conv_defn = self.params['conv_defn']
            if conv_defn == 'strict':
                self.eps = 1e-3
            elif conv_defn == 'normal':
                self.eps = 1e-2
        if 'stop_at_convergence' in self.params:
            stop_at_convergence = self.params['stop_at_convergence']
        else:
            stop_at_convergence = False
        if 'run_longer_at_convergence' in self.params:
            run_longer_at_convergence = self.params['run_longer_at_convergence']
        else:
            run_longer_at_convergence = False

        for it in range(max_iterations):
            print(f'Iteration {it}')
            self.save_state()

            sim = Pysim(
                root=self.root / f"iteration{it}",
                config=self.defaultsim.config,
                seqs=self.defaultsim.seqs,
                randomize_seed=True,
            )

            sim.set_chis(newchis)

            if self.lengthen_iterations and (it > 0):
                self.params["production_sweeps"] = int(
                    1.1 * self.params["production_sweeps"]
                )

            sim.run_eq(
                self.params["equilib_sweeps"],
                self.params["production_sweeps"],
                self.params["parallel"],
            )

            curr_chis = sim.flatten_chis()
            obs, jac = sim.load_observables(jacobian=True)

            gamma = self.params["gamma"]
            if self.dampen_first_step and (it == 0):
                gamma *= 0.25
            if self.dampen_first_step and (it == 1):
                gamma *= 0.25

            print(f"gammma = {gamma}")
            print("self.gamma = " + str(self.params["gamma"]))

            newchis, newloss = utils.newton(
                lam=obs,
                obj_goal=self.params["goals"],
                B=jac,
                gamma=gamma,
                current_chis=curr_chis,
                trust_region=self.params["trust_region"],
                method=self.params["method"],
            )

            converged = self.track_progress(newchis, newloss, sim)
            os.symlink(
                self.resources / "experimental_hic.npy",
                sim.root / "experimental_hic.npy",
            )

            self.analyze(sim.root)
            if converged:
                if stop_at_convergence:
                    break
                elif run_longer_at_convergence:
                    self.params["production_sweeps"] = self.final_it_sweeps

        if self.final_it_sweeps > 0:
            self.run_final_iteration(newchis)

        tf = time.time()
        return tf - t0

    def run_final_iteration(self, newchis):
        print(f'Final Iteration')
        self.save_state()

        # set up new config
        config = self.defaultsim.config.copy()
        sweeps = self.final_it_sweeps // self.params["parallel"]

        sim = Pysim(
            root=self.root / f"iteration{self.params['iterations']}",
            config=config,
            seqs=self.defaultsim.seqs,
            randomize_seed=True,
        )

        sim.set_chis(newchis)

        sim.run_eq(
            self.params["equilib_sweeps"],
            sweeps,
            self.params["parallel"],
        )

        os.symlink(
            self.resources / "experimental_hic.npy",
            sim.root / "experimental_hic.npy",
        )

        self.analyze(sim.root)


    def save_state(self):
        self_copy = copy.deepcopy(self)
        if (self_copy.resources / "experimental_hic.npy").exists():
            # save space, don't need to pickle gthic if it's already in resources
            del self_copy.gthic

        try:
            with open(self_copy.root / "backup.pickle", "wb") as f:
                pickle.dump(self_copy, f)
        except FileNotFoundError:
            with open("backup.pickle", "wb") as f:
                pickle.dump(self_copy, f)

    @classmethod
    def load_state(cls, filename: str):
        """loads maxent optimization from a saved state (pickle)
        reloads gthic, which is not included in pickle to save disk space
        """
        with open(filename, "rb") as f:
            loaded_maxent = pickle.load(f)

            if loaded_maxent.resources.exists():
                loaded_maxent.gthic = np.load(
                    loaded_maxent.resources / "experimental_hic.npy"
                )
            else:
                loaded_maxent.gthic = np.load("experimental_hic.npy")

            return loaded_maxent

    @classmethod
    def from_directory(cls, filename: PathLike):
        """loads maxent optimization from a directory
        reloads gthic, which is not included in pickle to save disk space
        """
        filename = Path(filename)
        with open(filename / "backup.pickle", "rb") as f:
            loaded_maxent = pickle.load(f)

            if loaded_maxent.resources.exists():
                loaded_maxent.gthic = np.load(
                    loaded_maxent.resources / "experimental_hic.npy"
                )
            else:
                loaded_maxent.gthic = np.load("experimental_hic.npy")

            return loaded_maxent

    def set_config(self, path):
        """load config from path"""
        self.config = utils.load_json(path)
