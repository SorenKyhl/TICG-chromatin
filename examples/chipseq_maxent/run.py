import sys
import numpy as np
from pathlib import Path
from pylib.datapipeline import DataPipeline
from pylib import epilib
from pylib.maxent import Maxent
import pylib.default 
from pylib import utils

encode_seqs = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3", "H3K9me3"]

def chip_maxent(chrom, wig_method, encode_only):
    res = 100_000
    size = 1024
    start = 0
    end = 120000000
    chrom = str(chrom)

    # load hic, sequences
    pipe = DataPipeline(res, chrom, start, end, size)

    hicdata_dir = Path("/home/skyhl/Documents/chromatin/hic-data/HCT116_auxin/")
    gthic = pipe.load_hic(hicdata_dir/"HIC-GSE104333_Rao-2017-treated_6hr_combined_30.hic")
    
    chipseqdata_dir = Path("/home/skyhl/Documents/chromatin/maxent-analysis/HCT116_chipseq_hg19/")
    HCT116_bigWig = list(chipseqdata_dir.glob("*.bigWig"))
    sequences = pipe.load_chipseq_from_files(HCT116_bigWig, wig_method)

    if encode_only:
        seqs = []
        for name in encode_seqs:
            seqs.append(sequences[name])
        sequences = np.array(seqs)
        print("encode loaded")
        print(np.shape(sequences))
    else:
        sequences = np.array(list(sequences.values()))
    sequences = np.array([pylib.default.chipseq_pipeline.fit(seq) for seq in sequences])

    # set up config, params, goals
    #config = pylib.default.config
    config = utils.load_json("config.json")
    goals = epilib.get_goals(gthic, sequences, config['beadvol'], config['grid_size'])

    params = pylib.default.params
    params['equilib_sweeps'] = 10000
    params['production_sweeps'] = 50000
    params['trust_region'] = 1000
    params['parallel'] = 7
    params['iterations'] = 10
    params['goals'] = goals

    # initial optimization
    me_name = Path("encode")
    me = Maxent(me_name, params, config, sequences, gthic)
    me.fit()

    # final optimization
    me.params["production_sweeps"] = 400000 
    me.params["iterations"] = 10
    me.set_root("encode-cont")
    me.initial_chis = me.chis[-1]
    me.dampen_first_step = False
    me.fit()

def main():
    """
    for chrom in range(1,8):
        for wig_method in ["mean", "max"]:
            chip_maxent(chrom, wig_method)
            """
    chip_maxent(str(2), "max", True)

if __name__ == "__main__":
    main()

