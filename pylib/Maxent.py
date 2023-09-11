import copy
import logging
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
        plaid_diagonly: bool = False,
        norm: bool = False,
        fast_analysis: bool = False,
        mkdir: bool = True,
        bound_diag_chis: bool = False
    ):
        """
        root: root of maxent filesystem
        params: maxent parameters
        config: default simulation config
        seqs: simulation sequences
        gthic: ground truth hic
        overwrite: will overwrite existing files
        bound_diag_chis: will ensure that diag chis start at 0
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
        self.bound_diag_chis = bound_diag_chis

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
        self.plaid_diagonly = plaid_diagonly

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
        if self.k > 0:
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

        if plaid is not None:
            self.track_plaid_chis = np.vstack((self.track_plaid_chis, plaid))
            np.savetxt(self.root / "chis.txt", plaid, fmt="%.4f", newline=" ")
            np.save(self.root / "plaid_chis.npy", self.track_plaid_chis)
            self.plot_plaid_chis()

        self.track_diag_chis = np.vstack((self.track_diag_chis, diag))
        np.savetxt(self.root / "chis_diag.txt", diag, fmt="%.4f", newline=" ")
        np.save(self.root / "diag_chis.npy", self.track_diag_chis)
        self.plot_diag_chis()

        np.savetxt(self.root / "convergence.txt", self.loss, fmt="%.16f")
        converged = self.plot_convergence()

        return converged

    def plot_convergence(self):
        max_it = len(self.loss)
        iterations = np.arange(1, max_it+1)
        if max_it < 2:
            return False

        # plot loss
        plt.figure()
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.plot(iterations, self.loss, ".-")

        # convergence based on loss
        converged_it = None
        for i in range(1, max_it):
            diff = self.loss[i] - self.loss[i-1]
            if np.abs(diff) < self.eps and self.loss[i] < self.loss[0]:
                converged_it = iterations[i]
                break

        if converged_it is not None and max_it > 2:
            plt.axvline(converged_it, color = 'k', label = 'converged')
            plt.legend()
            converged = True
        else:
            converged = False

        plt.tight_layout()
        plt.savefig(self.root / "loss.png")
        plt.close()

        # convergence in parameter space
        convergence = []
        for j in range(1, max_it+1):
            diff = self.chis[j] - self.chis[j-1]
            conv = np.linalg.norm(diff, ord = 2)
            convergence.append(conv)
        print(convergence)
        plt.plot(iterations, convergence)
        plt.yscale('log')
        plt.axhline(100, c='k', ls='--')
        plt.axhline(10, c='k', ls='--')
        plt.ylabel(r'$|\chi_{i}-\chi_{i-1}|$ (L2 norm)', fontsize=16)
        plt.xlabel('Iteration', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.root / "param_convergence.png")
        plt.close()

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
        plt.tight_layout()
        if legend:
            plt.legend(loc=(1.04,0), ncol = 3)
            plt.savefig(self.root / "track_plaid_chis_legend.png")
        else:
            plt.savefig(self.root / "track_plaid_chis.png")
        plt.close()

    def plot_diag_chis(self):
        plt.figure()
        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Diagonal Parameter Value", fontsize=16)
        plt.plot(self.track_diag_chis, ".-")
        plt.tight_layout()
        plt.savefig(self.root / "track_diag_chis.png")
        plt.close()

    def analyze(self, dir):
        if self.analysis_on:
            analysis.main(dir=dir)

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
            obj_goal = np.array(self.params["goals"])


            if self.plaid_diagonly:
                inds = self.plaid_diagonly_inds(sim)
                new_chis_diagonly = np.zeros_like(curr_chis)
                obs = obs[inds]
                obj_goal = obj_goal[inds]
                jac = jac[np.ix_(inds, inds)]
                curr_chis = curr_chis[inds]

            gamma = self.params["gamma"]
            if self.dampen_first_step and (it == 0):
                gamma *= 0.25
            if self.dampen_first_step and (it == 1):
                gamma *= 0.25

            print(f"gammma = {gamma}")
            print("self.gamma = " + str(self.params["gamma"]))

            newchis, newloss = utils.newton(
                lam=obs,
                obj_goal=obj_goal,
                B=jac,
                gamma=gamma,
                current_chis=curr_chis,
                trust_region=self.params["trust_region"],
                method=self.params["method"],
            )

            if self.bound_diag_chis:
                plaid, diag = sim.split_chis(newchis) # these are a view (reference type)
                assert diag[0] < 1e-5, f'diag[0] = {diag[0]}'
                diag -= diag[1] # push diag chis down s.t. first diag chi is zero
                diag[0] = 0 # zeroth diag chi should stay at zero


            if self.plaid_diagonly:
                new_chis_diagonly[inds] = newchis
                newchis = new_chis_diagonly

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

    def plaid_diagonly_inds(self, sim):
        """get indices for chis and observables to keep when using plaid diagonly."""

        # when plaid_chis are flattened, get indices for plaid chis along the main diagonal
        #plaid_chis_diag_inds = np.diag_indices_from(plaid_chis)
        #plaid_chis_diag_inds_flat = np.ravel_multi_index(plaid_chis_diag_inds, np.shape(plaid_chis))

        # raster flat indices into upper triangular matrix, then get just the diagonal entries
        plaid_chis_flat, diag_chis_flat = sim.split_chis(sim.flatten_chis())
        n = len(sim.config["chis"])
        tri = np.zeros_like(sim.config["chis"])
        flat_indices = list(range(len(plaid_chis_flat)))
        tri[np.triu_indices(n)] = flat_indices
        plaid_chis_diag_inds_flat = np.diagonal(tri)
        plaid_chis_diag_inds_flat = np.array(plaid_chis_diag_inds_flat, dtype=int)

        # when allchis are flattened, get indices for all diagonal chis
        ndiag_chis = len(sim.config["diag_chis"])
        start = plaid_chis_diag_inds_flat[-1] + 1
        diag_chis_inds = np.arange(start, start+ndiag_chis)

        inds = np.hstack((plaid_chis_diag_inds_flat, diag_chis_inds))
        return inds
