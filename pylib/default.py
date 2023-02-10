from pathlib import Path
from pylib import utils, hic
from pylib.chipseqPipeline import ChipseqPipeline, Smooth, Normalize, Sigmoid
from pylib.datapipeline import DataPipeline

""" 
contains default config and params files 
"""

proj_root = Path("/home/skyhl/Documents/TICG-chromatin")
config = utils.load_json(proj_root/"maxent/defaults/config.json")
params = utils.load_json(proj_root/"maxent/defaults/params.json")
chipseq_pipeline = ChipseqPipeline([Smooth(), Normalize(), Sigmoid()])
res = 100000
chrom = 2 
start = 0 
end = 120_000_000
size = 1024
data_pipeline = DataPipeline(res, chrom, start, end, size)

HCT116_hic = "/home/skyhl/Documents/chromatin/hic-data/HCT116_auxin/HIC-GSE104333_Rao-2017-treated_6hr_combined_30.hic"
HCT116_chipseq = "/home/skyhl/Documents/chromatin/maxent-analysis/HCT116_chipseq_hg19/"

data_dir = Path("data")
HCT116_hic_20k = data_dir/"HCT116_chr2_20k.npy"

def get_hic(nbeads, pool_fn = hic.pool):
    if not data_dir.exists():
        data_dir.mkdir()

    nbeads_large = 20480
    pipe = copy.deepcopy(data_pipeline)
    pipe.resize(nbeads_large)
    gthic = pipe.get_hic(HCT116_hic)
    
    if not HCT116_hic_20k.exists():
        np.save(HCT116_hic_20k, gthic)

    factor = int(nbeads_large/nbeads)
    return pool_fn(nbeads, factor)
    





