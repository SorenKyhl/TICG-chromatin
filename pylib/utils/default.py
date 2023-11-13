import os
import os.path as osp
import sys
from pathlib import Path

from pylib.chipseqPipeline import ChipseqPipeline, Normalize, Sigmoid, Smooth
from pylib.datapipeline import DataPipeline
from pylib.utils import utils

"""
contains default config and params files
"""

usr = sys.path[1].split(os.sep)[2]
if usr == 'skyhl':
    root = "/home/skyhl/Documents/"
elif usr == 'erschultz':
    root = "/home/erschultz"
else:
    raise Exception(f"Unrecognized user: {sys.path}")
proj_root = Path(root, "TICG-chromatin")
config = utils.load_json(proj_root / f"defaults/config_{usr}.json")
params = utils.load_json(proj_root / f"defaults/params_{usr}.json")


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

data_dir = proj_root / "data"
HCT116_hic_20k = {'2': data_dir / "HCT116_chr2_20k.npy",
                  '10': data_dir/ "HCT116_chr10_20k.npy"}
HCT116_seqs_20k = data_dir / "seqs20k.npy"
