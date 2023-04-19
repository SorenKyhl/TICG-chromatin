import os.path as osp
from pathlib import Path

from pylib import utils
from pylib.chipseqPipeline import ChipseqPipeline, Normalize, Sigmoid, Smooth
from pylib.datapipeline import DataPipeline

"""
contains default config and params files
"""

root = "/home/skyhl/Documents/"
eric = False
if not osp.exists(root):
    eric = True
    root = "/home/erschultz"
proj_root = Path(root, "TICG-chromatin")
if eric:
    config = utils.load_json(proj_root / "utils/default_config.json")
else:
    config = utils.load_json(proj_root / "maxent/defaults/config.json")
params = utils.load_json(proj_root / "maxent/defaults/params.json")
chipseq_pipeline = ChipseqPipeline([Smooth(), Normalize(), Sigmoid()])
res = 100000
chrom = 2
start = 0
end = 120_000_000
size = 1024
data_pipeline = DataPipeline(res, chrom, start, end, size)

hic_paths = {
        "HCT116_auxin" : f"{root}/chromatin/hic-data/HCT116_auxin/HIC-GSE104333_Rao-2017-treated_6hr_combined_30.hic",
        "GM12878" : f"{root}/chromatin/hic-data/GM12878/GSE63525_GM12878_insitu_replicate_30.hic"
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
