import shutil
import time
from multiprocessing import Process
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from pylib.pyticg import Sim
from pylib.utils import utils
from pylib.utils.utils import cat, cd, copy_last_snapshot

PathLike = Union[Path, str]
ArrayLike = Union[list, np.ndarray]

"""
pysim
"""


class Pysim:
    """wrapper around pyticg"""

    def __init__(
        self,
        root: PathLike,
        config: dict,
        seqs: Optional[ArrayLike],
        gthic=None,
        randomize_seed: bool = True,
        mkdir: bool = True,
        setup_needed: bool = True,
        overwrite: Optional[bool] = False,
        umatrix: Optional[ArrayLike] = None,
        lmatrix: Optional[ArrayLike] = None,
        dmatrix: Optional[ArrayLike] = None

    ):
        self.set_root(root, mkdir, overwrite)
        self.set_config(config)
        if seqs is not None and seqs.shape[0] > seqs.shape[1]:
            # Pysim assumes seq is kxm
            self.seqs = seqs.T
        else:
            self.seqs = seqs
        self.U = umatrix
        self.L = lmatrix
        self.D = dmatrix

        self.setup_needed = setup_needed
        # should be true unless instantiated using from_directory

        if gthic is not None:
            np.save(self.root / "experimental_hic.npy", gthic)

        if randomize_seed:
            self.randomize_seed()

    @classmethod
    def from_directory(cls, sim_dir: PathLike, new_root: Optional[PathLike] = None):
        """construct a simulation object from a directory that's already set up
        (i.e. contains config and sequence files)

        dir: simulation directory from which to initialize
        root: new simulation root directory
        """
        sim_dir = Path(sim_dir).absolute()
        config = utils.load_json(sim_dir / "config.json")
        with utils.cd(sim_dir):
            seqs = utils.load_sequences(config)

        if new_root is None:
            return cls(sim_dir, config, seqs, mkdir=False, setup_needed=False)
        else:
            return cls(new_root, config, seqs, mkdir=True, setup_needed=True)

    def set_root(self, root: PathLike, mkdir: bool, overwrite: Optional[bool]=False):
        """set the root of the simulation. and (optionally) create the directory

        Args:
            root: simulation root, where all simulation outputs will be stored
            mkdir: if true, will make a new directory at the root path
        """

        self.root = Path(root)
        if overwrite and self.root.exists():
            shutil.rmtree(self.root)
        if mkdir:
            self.root.mkdir(exist_ok=False)

    def set_config(self, config: Union[dict, str]):
        """set the config, either by assignment or by loading from a json file"""
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = utils.load_json(config)

    def randomize_seed(self):
        """sets seed for random number generator used in Monte Carlo simulation"""
        self.config["seed"] = np.random.randint(10000)

    def setup(self):
        """write simulation inputs in simulation root directory, if setup_needed"""
        # needs to happen regardless of setup_needed setting, because seed is randomized
        utils.write_json(self.config, Path(self.root, "config.json"))

        # further setup needed - write sequences to directory
        if self.setup_needed:
            if self.seqs is not None:
                self.seqs = np.array(self.seqs)
                if self.seqs.ndim > 1:
                    for i, seq in enumerate(self.seqs):
                        self.write_sequence(seq, Path(self.root, f"pcf{i+1}.txt"))
                else:
                    self.write_sequence(self.seqs, Path(self.root, "pcf1.txt"))
                np.save(Path(self.root, f"x.npy"), self.seqs)
            if self.U is not None:
                np.save(Path(self.root, f"U.npy"), self.U)
                np.savetxt(Path(self.root, f"umatrix.txt"), self.U)
            if self.L is not None:
                np.save(Path(self.root, f"L.npy"), self.L)
                np.savetxt(Path(self.root, f"lmatrix.txt"), self.L)
            if self.D is not None:
                np.save(Path(self.root, f"D.npy"), self.D)
                np.savetxt(Path(self.root, f"dmatrix.txt"), self.D)


    def flatten_chis(self):
        """restructure chi parameters into a 1-dimensional list
        Returns: [1 x n] list = [plaid_chis, diag_chis]
        """
        indices = np.triu_indices(self.config["nspecies"])
        diag_chis = np.array(self.config["diag_chis"])
        if "chis" in self.config:
            plaid_chis = np.array(self.config["chis"])[indices]
            flat_chis = np.hstack((plaid_chis, diag_chis))
        else:
            flat_chis = diag_chis
        return flat_chis

    def chis_to_matrix(self, flat_chis):
        """restructure 1-D list of plaid chis into matrix
        Returns: [n x n] chi matrix
        """
        nspecies = self.config["nspecies"]
        X = np.zeros((nspecies, nspecies))
        X[np.triu_indices(nspecies)] = flat_chis
        X = X + X.T - np.diag(np.diag(X))
        return X

    def split_chis(self, allchis):
        """splits 1-D list of chis into: plaid_chis, diag_chis"""
        nplaidchis = len(allchis) - len(self.config["diag_chis"])
        if nplaidchis > 0:
            plaid_chis_flat, diag_chis = np.split(allchis, [nplaidchis])
            return plaid_chis_flat, diag_chis
        else:
            return None, allchis

    def set_chis(self, allchis):
        """takes 1d vector of all chis and updates config chi parameters"""
        plaid_chis_flat, diag_chis = self.split_chis(allchis)
        if plaid_chis_flat is not None:
            self.config["chis"] = self.chis_to_matrix(plaid_chis_flat).tolist()
        self.config["diag_chis"] = diag_chis.tolist()

    def load_observables(self, jacobian=False, mode='all'):
        """load observable trajectories from simulation output.
        return mean of observables throughout the simulation,
        and (optionally) the jacobian of the observable matrix
        """
        if self.data_out is None:
            raise ValueError("data out member variable has not been initialized")

        obs_files = []
        if mode == 'all':
            obs_files.append(self.root / self.data_out / "observables.traj")
            obs_files.append(self.root / self.data_out / "diag_observables.traj")
        elif mode == 'diag':
            obs_files.append(self.root / self.data_out / "diag_observables.traj")
        elif mode == 'plaid':
            obs_files.append(self.root / self.data_out / "observables.traj")
        else:
            raise Exception(f'Unrecognized mode: {mode}')

        df_total = pd.DataFrame()
        for file in obs_files:
            df = pd.read_csv(file, delimiter="\t", header=None)
            df = df.dropna(axis=1)
            df = df.drop(df.columns[0], axis=1)
            df_total = pd.concat((df_total, df), axis=1)

        self.obs = df_total.mean().values

        if jacobian:
            self.jac = df_total.cov().values  # pyright: ignore
            return self.obs, self.jac
        else:
            return self.obs

    def run(self, name=None):
        """run simulation"""
        t0 = time.time()
        self.data_out = name
        self.setup()
        with cd(self.root):
            if name:
                engine = Sim(self.data_out)  # output to dir: name
            else:
                engine = Sim()  # output to dir: data_out
                self.data_out = "data_out"  # don't set earlier...
                # in this case, use default constructor so engine.run() pipes to stdout
            engine.run()
        tf = time.time()
        return tf - t0

    def run_eq(
        self,
        equilibrium_sweeps: int,
        production_sweeps: int,
        parallel_simulations: int = 1,
    ):
        """run equilibration followed by production simulation
        the production run can be executed in parallel.

        Args:
            equilibrium_sweeps: number of equilibrium simulation sweeps
            production_sweeps: number of production simulation sweeps (per core)
            parallel_simulations: number of parallel production simulations to execute
        """
        t0 = time.time()
        equilibration_dir = "equilibration"
        production_dir = "production_out"

        # equilibration
        self.config["nSweeps"] = equilibrium_sweeps
        self.config["load_configuration"] = False
        t_eq = self.run(equilibration_dir)
        print(f'Equillibration took {np.round(t_eq, 2)} seconds')

        # production. copy the last structure to initialize production simulations
        equilibrated_structure = "equilibrated.xyz"
        copy_last_snapshot(
            xyz_in=(self.root / equilibration_dir / "output.xyz"),
            xyz_out=(self.root / equilibrated_structure),
            nbeads=self.config["nbeads"],
        )

        self.config["nSweeps"] = production_sweeps
        self.config["load_configuration"] = True
        self.config["load_configuration_filename"] = equilibrated_structure

        if parallel_simulations == 1:
            t_prod = self.run(production_dir)
        else:
            print(f'Running in parallel with {parallel_simulations} runs')
            t_prod = self.run_parallel(production_dir, parallel_simulations)
        print(f'Production took {np.round(t_prod, 2)} seconds')

        tf = time.time()
        return tf - t0

    def run_parallel(self, name: str, cores: int):
        """run production, using parallel simulations with different initial seeds

        each parallel core dumps output to their own directory. after completion,
        the individual simulation data are aggregated into a final production directory

        Args:
            name: name of output file for aggregated simulation data
            cores: number of parallel simulations to execute
        """
        t0 = time.time()
        self.data_out = name
        processes = []
        # create processes
        for i in range(cores):
            target = self.run
            args = (f"core{i}",)
            processes.append(Process(target=target, args=args))

        # run simulations
        for p in processes:
            self.randomize_seed()
            p.start()  # run simulation for process p
            sleeptime = int(
                max(1, self.config["nbeads"] / 1000)
            )  # larger simulations take longer to move over
            time.sleep(sleeptime)  # engine needs time to load config.json before next iteration

        # wait to join
        for p in processes:
            p.join()

        # getting segfaults, possibly due to all processes retaining their memory. does this fix it?
        for p in processes:
            p.terminate()

        self.aggregate_production_files()
        tf = time.time()
        return tf - t0

    def aggregate_production_files(self):
        """aggregate simulation data from each core into final production folder.

        each parallel core dumps its output to a separate folder
        after all simulations are finished...
        - concatenate all observables and energies into one trajectory file
        - sum all contact maps to produce one aggregate contact map
        """
        Path(self.root / "production_out").mkdir()

        # aggregate files are concatenated together to form one trajectory
        aggregate_files = ["observables.traj",
                            "diag_observables.traj",
                            "output.xyz"]
        for file in aggregate_files:
            cat(
                self.root / "production_out" / file,
                self.root.glob("core*/" + str(file)),
            )

        cat(
            self.root / "production_out" / "energy.traj",
            self.root.glob("core*/energy.traj"),
            True
            )


        # contact maps are summed together to form one contact map
        contact_files = list(self.root.glob("core*/contacts.txt"))
        self.combine_contactmaps(
            contact_files, output_file=self.root / "production_out/contacts.txt"
        )

        production = Pysim(
            self.root / "production_out", self.config, self.seqs, mkdir=False
        )
        production.setup()  # save config, seqs to production_out

        # delete redundant contact files. these have been aggregated into production/contacts.txt
        [contact_file.unlink(missing_ok=False) for contact_file in contact_files]

    def combine_contactmaps(
        self, contact_files: Sequence[PathLike], output_file: Optional[PathLike] = None
    ):
        """combines (sums) multiple contact maps from separate parallel simulations

        Args:
            contact_files: list of files containing contact maps to be combined
            output_file: name for output combined contact map.
        Returns:
            contact map, or saves contact map to file if specified
        """
        combined = np.loadtxt(contact_files[0])
        for file in contact_files[1:]:
            combined += np.array(
                pd.read_csv(file, header=None, sep=" ")
            )  # much faster than np.loadtxt
            # combined += np.loadtxt(file)

        if output_file:
            np.savetxt(output_file, combined, fmt="%d", delimiter=" ")
        else:
            return combined

    def write_sequence(self, sequence: ArrayLike, path: PathLike):
        """write sequence of polymer bead types to disk at specified path"""
        np.savetxt(path, sequence, fmt="%.8f", delimiter=" ")
