import numpy as np
import pandas as pd
from pathlib import Path
import pyBigWig
import hicstraw

from pylib import epilib

class DataPipeline:
    def __init__(self, res, chrom, start, end, size):
        self.res = res
        self.chrom = str(chrom)
        self.chromstr = "chr"+str(self.chrom)
        self.start = start
        self.end = end
        self.size = size
        
    def load_hic(self, filename):
        hic = hicstraw.HiCFile(filename)
        contact = epilib.load_contactmap_hicstraw(hic, self.res, self.chrom, self.start, self.end)
        self.bigsize, _ = np.shape(contact)
        contact, self.dropped_inds = epilib.clean_contactmap(contact)
        contact = epilib.get_contactmap(contact, normtype="mean") #TODO- phase this out into it's own function
        return contact[0:self.size,0:self.size]
        
    def load_bigWig(self, filename, method="mean"):
        """
        load chipseq from .bigWig file format using pyBigWig
        can load from local or remote files.
        """
        bw = pyBigWig.open(str(filename)) # H3K36me3 FC
        signal = bw.stats(self.chromstr, self.start, self.end, type=method, nBins=self.bigsize)
        signal = np.delete(signal, self.dropped_inds)
        return np.array(signal[0:self.size])
    
    def load_wig(self, filename, method):
        """
        load chipseq from .wig file format using custom routines
        can only load from local files.
        """
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
