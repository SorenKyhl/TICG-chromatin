"""
Contains default config and params files.
"""
import os
import sys
from pathlib import Path

from pylib.chipseqPipeline import ChipseqPipeline, Normalize, Sigmoid, Smooth
from pylib.datapipeline import DataPipeline
from pylib.utils import utils

usr = sys.path[0].split(os.sep)[2]
found_root = False
if usr == 'skyhl':
    root = "/home/skyhl/Documents/"
    found_root = True
else:
    print(f"pylib.utils.default user not set up: {sys.path[0]}")

if found_root:
    proj_root = Path(root, "TICG-chromatin")
    config = utils.load_json(proj_root / f"defaults/config_{usr}.json")
    params = utils.load_json(proj_root / f"defaults/params_{usr}.json")

    data_dir = proj_root / "data"
    HCT116_hic_20k = {'2': data_dir / "HCT116_chr2_20k.npy",
                      '10': data_dir/ "HCT116_chr10_20k.npy"}
    HCT116_seqs_20k = data_dir / "seqs20k.npy"

    hic_paths = {
            "HCT116_auxin" : f"{root}/chromatin/hic-data/HCT116_auxin/HIC-GSE104333_Rao-2017-treated_6hr_combined_30.hic",
            "GM12878" : f"{root}/chromatin/hic-data/GM12878/GSE63525_GM12878_insitu_replicate_30.hic",
            "IMR90": "/mnt/hic-data/IMR90/GSE63525_IMR90_combined_30.hic",
            "IMR90-new": "/mnt/hic-data/IMR90/ENCFF946RTR.hic"
                }

    HCT116_hic = Path(
        f"{root}/chromatin/hic-data/HCT116_auxin/HIC-GSE104333_Rao-2017-treated_6hr_combined_30.hic"
    )
    HCT116_chipseq = Path(
        f"{root}/chromatin/maxent-analysis/HCT116_chipseq_hg19/"
    )
else:
    params = {
        "iterations": 20,
        "gamma": 1,
        "trust_region": None,
        "equilib_sweeps": 10000,
        "production_sweeps": 300000,
        "goal_specified": 1,
        "overwrite": 0,
        "parallel": 1,
        "randomize_seed": 1,
        "mode": "all", # optimizaing all parameters
        "method": "n", # Newton's method
        "run_longer_at_convergence": False
    }

    config = {
        "conservative_contact_pooling": False,
        "double_count_main_diagonal": False,
        "angles_on": False,
        "k_angle": 0,
        "theta_0": 180,
        "cell_volumes":  False,
        "compressibility_on": False,
        "diag_pseudobeads_on": True,
        "parallel": False,
        "boundary_type": "spherical",
        "profiling_on": False,
        "print_acceptance_rates": True,
        "load_bead_types": True,
        "load_configuration": False,
        "contact_resolution": 1,
        "track_contactmap": False,
        "diagonal_linear": True,
        "dump_density": False,
        "dump_observables": False,
        "update_contacts_distance": False,
        "aspect_ratio": 1.0,

        "lmatrix_on": True,
        "umatrix_on": True,
        "dmatrix_on": True,

        "nSweeps": 300000,
        "dump_frequency": 100000,
        "dump_stats_frequency": 10,

        "bonded_on": True,
        "nonbonded_on": True,
        "plaid_on": True,
        "diagonal_on": False,
        "diag_start": 0,


        "displacement_on": False,
        "translation_on": True,
        "crankshaft_on": True,
        "pivot_on": True,
        "rotate_on": False,
        "gridmove_on": True,

        "density_cap_on": True,
        "phi_solvent_max": 0.5,

        "bond_type": "gaussian"
    }

bonded_config = config.copy()
bonded_config['nonbonded_on'] = False
bonded_config['plaid_on'] = False
bonded_config['diagonal_on'] = False
bonded_config['gridmove_on'] = False
bonded_config['grid_size'] = 100 # irrelevant
bonded_config["nSweeps"] = 50000
bonded_config["dump_frequency"] = 1000
bonded_config["dump_stats_frequency"] = 1000
bonded_config['lmatrix_on'] = False
bonded_config['dmatrix_on'] = False
bonded_config['smatrix_on'] = False

chipseq_pipeline = ChipseqPipeline([Smooth(), Normalize(), Sigmoid()])
res = 100000
chrom = 2
start = 0
end = 120_000_000
size = 1024
data_pipeline = DataPipeline(res, chrom, start, end, size)
