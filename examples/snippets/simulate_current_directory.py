import sys
from pylib.pysim import Pysim

if __name__ == "__main__":
    sim = Pysim.from_directory(".")
    if len(sys.argv) == 1:
        sim.run_eq(10000, 50000, 7)
    if len(sys.argv) == 2:
        sim.run_eq(10000, sys.argv[1], 7)

