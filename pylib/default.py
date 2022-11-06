from pathlib import Path
from pylib import utils
from pylib.chipseqPipeline import ChipseqPipeline, Smooth, Normalize, Sigmoid

""" 
contains default config and params files 
"""

proj_root = Path("/home/skyhl/Documents/TICG-chromatin")
config = utils.load_json(proj_root/"maxent/defaults/config.json")
params = utils.load_json(proj_root/"maxent/defaults/params.json")
chipseq_pipeline = ChipseqPipeline([Smooth(), Normalize(), Sigmoid()])

