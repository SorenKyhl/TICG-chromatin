import numpy as np
import pandas as pd
from pathlib import Path
import pyBigWig
import hicstraw

from pylib import epilib

class DataPipeline:
    def __init__(self, res, chrom, start, end, size):
        self.res = int(res)
        self.chrom = str(chrom)
        self.chromstr = "chr"+str(self.chrom)
        self.start = start
        self.end = end
        self.size = size
        
    def load_hic(self, filename, KR=True, clean=True, rescale_method="mean"):
        """
        KR: bool, knight-rubin normalization.
        rescale_method: str ["mean", "max"], rescale contactmap so entries are probabilities, rather than frequencies
        clean: remove rows and colums for which the main diagonal is zero
        """
        filename = str(filename) # hicstraw doesn't accept pathlib objects
        hic = hicstraw.HiCFile(filename)
        contactmap = epilib.load_contactmap_hicstraw(hic, self.res, self.chrom, self.start, self.end, KR=KR)

        self.bigsize, _ = np.shape(contactmap) # used for loading the correct number of chipseq bins

        if clean:
            contactmap, self.dropped_inds = epilib.clean_contactmap(contactmap)

        if rescale_method:
            contactmap = epilib.rescale_contactmap(contactmap, method=rescale_method) 

        return contactmap[0:self.size,0:self.size]
        
    def load_bigWig(self, filename, method="mean"):
        """
        load chipseq from .bigWig file format using pyBigWig
        can load from local or remote files.
        """
        assert(method in ["mean", "max"])
        filename = str(filename) # pyBigWig doesn't accept pathlib objects
        bw = pyBigWig.open(filename)
        signal = bw.stats(self.chromstr, self.start, self.end, type=method, nBins=self.bigsize)
        signal = np.delete(signal, self.dropped_inds)
        return np.array(signal[0:self.size])
    
    def load_wig(self, filename, method):
        """
        load chipseq from .wig file format using custom routines
        can only load from local files.
        """
        assert(method in ["mean", "max"])
        df = pd.read_csv(filename, sep='\t', names=['start','end','value'], skiprows=1)
        chip = epilib.bin_chipseq(df, res, method=method) # this also sets the baseline for "low signal"
        chip = np.nan_to_num(chip)
        return chip
    
    def load_chipseq_from_files(self, filenames, method):
        """
        filenames: list of paths to chipseq files (.bigWig or .wig)
        method: (str) ["mean", "max"] - method to call on each bin of data
        """
        seqs = {}
        for file in filenames:
            extension = file.suffix

            if extension == '.bigWig':
                name = file.name.split('_')[0]
                seqs[name] = self.load_bigWig(file, method)
            elif extension == '.wig':
                name = str(file).split("_")[-2]
                seqs[name] = self.load_wig(file, method)
            else:
                print("file extension must either be .bigWig or .wig")
                raise ValueError

        return seqs
