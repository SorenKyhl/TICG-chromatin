import numpy as np
import pandas as pd
from pathlib import Path
import pyBigWig
import hicstraw

from pylib import epilib
from pylib import hic as hiclib


def get_experiment_marks(directory):
    """
    make mapping between experimental codes and epigenetic marks in specified directory

    Args:
        (str) directory: directory containing .bigwig files
    Returns:
        (dict) lookup_table[str, str] maps experiment codes to epigenetic marks
    """

    directory = Path(directory)
    metadata = pd.read_csv(directory / "metadata.tsv", sep="\t")
    marks = metadata["Experiment target"].apply(lambda s: s.split("-")[0])
    lookup_table = dict(zip(metadata["File accession"], marks))
    return lookup_table


class DataPipeline:
    """class for loading and manipulating hic and chipseq files

    Args:
        res: resolution of contact map in base pairs per bin
        chrom: chromosome
        start: lower bound in base pairs
        end: upper bound in base pairs
        size: desired number of bins along one dimension of contactmap
    """

    def __init__(self, res, chrom, start, end, size):
        self.res = int(res)
        self.start = start
        self.end = end
        self.size = size
        self.set_chrom(chrom)

        self.bigsize = self.size
        self.dropped_inds = []

    def set_chrom(self, chrom):
        self.chrom = str(chrom)
        self.chromstr = "chr" + str(self.chrom)

    def resize(self, newsize):
        factor = int(newsize / self.size)
        self.res = int(self.res / factor)
        self.size = newsize
        return self

    def load_hic(self, filename, KR=True, clean=True, rescale_method="ones"):
        """load hic from .hic file
        KR: bool, knight-rubin normalization.
        rescale_method: str ["mean", "ones"], rescale contactmap to valid probabilities
        clean: remove rows and colums for which the main diagonal is zero
        """
        filename = str(filename)  # hicstraw doesn't accept pathlib objects
        hic = hicstraw.HiCFile(filename)
        contactmap = epilib.load_contactmap_hicstraw(
            hic, self.res, self.chrom, self.start, self.end, KR=KR
        )

        self.bigsize, _ = np.shape(
            contactmap
        )  # used for loading the correct number of chipseq bins

        if clean:
            contactmap, self.dropped_inds = epilib.clean_contactmap(contactmap)

        # set main diag to one (on average)
        if rescale_method:
            contactmap = hiclib.normalize_hic(contactmap, rescale_method)
            #raise ValueError("deprecated")
            # contactmap = epilib.rescale_contactmap(contactmap, method=rescale_method)

        return contactmap[0 : self.size, 0 : self.size]

    def load_bigWig(self, filename, method="mean"):
        """
        can load from local or remote files.
        """
        assert method in ["mean", "max"]
        filename = str(filename)  # pyBigWig doesn't accept pathlib objects
        bw = pyBigWig.open(filename)
        signal = bw.stats(
            self.chromstr, self.start, self.end, type=method, nBins=self.bigsize
        )
        signal = np.delete(signal, self.dropped_inds)
        return np.array(signal[0 : self.size])

    def load_wig(self, filename, method):
        """
        load chipseq from .wig file format using custom routines
        can only load from local files.
        """
        assert method in ["mean", "max"]
        df = pd.read_csv(
            filename, sep="\t", names=["start", "end", "value"], skiprows=1
        )
        chip = epilib.bin_chipseq(
            df, self.res, method=method
        )  # this also sets the baseline for "low signal"
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

            if extension == ".bigWig":
                name = file.name.split("_")[0]
                seqs[name] = self.load_bigWig(file, method)
            elif extension == ".wig":
                name = str(file).split("_")[-2]
                seqs[name] = self.load_wig(file, method)
            else:
                print("file extension must either be .bigWig or .wig")
                raise ValueError


    def load_chipseq_from_directory(self, directory, method):
        """
        filenames: list of paths to chipseq files (.bigWig or .wig)
        method: (str) ["mean", "max"] - method to call on each bin of data
        """
        seqs = {}
        directory = Path(directory)
        filenames = list(directory.glob("*.bigWig"))
        lookup_table = get_experiment_marks(directory)

        for file in filenames:
            extension = file.suffix

            if extension == ".bigWig":
                name = lookup_table[file.stem]
                seqs[name] = self.load_bigWig(file, method)
            elif extension == ".wig":
                name = lookup_table[file.stem]
                seqs[name] = self.load_wig(file, method)
            else:
                print("file extension must either be .bigWig or .wig")
                raise ValueError
            print(f'loaded {name}')

        return seqs
