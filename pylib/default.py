from pathlib import Path
from pylib import utils
from pylib.chipseqPipeline import ChipseqPipeline, Smooth, Normalize, Sigmoid
from pylib.datapipeline import DataPipeline

"""
contains default config and params files
"""

proj_root = Path("/home/skyhl/Documents/TICG-chromatin")
config = utils.load_json(proj_root / "maxent/defaults/config.json")
params = utils.load_json(proj_root / "maxent/defaults/params.json")
chipseq_pipeline = ChipseqPipeline([Smooth(), Normalize(), Sigmoid()])
res = 100000
chrom = 2
start = 0
end = 120_000_000
size = 1024
data_pipeline = DataPipeline(res, chrom, start, end, size)

HCT116_hic = Path(
    "/home/skyhl/Documents/chromatin/hic-data/HCT116_auxin/HIC-GSE104333_Rao-2017-treated_6hr_combined_30.hic"
)
HCT116_chipseq = Path(
    "/home/skyhl/Documents/chromatin/maxent-analysis/HCT116_chipseq_hg19/"
)

data_dir = proj_root / "data"
HCT116_hic_20k = data_dir / "HCT116_chr2_20k.npy"
HCT116_seqs_20k = data_dir / "seqs20k.npy"
