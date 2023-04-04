import sys
import logging
from pylib.pysim import Pysim

if __name__ == "__main__":
    sim = Pysim.from_directory(".")
    if len(sys.argv) == 1:
        sim.run_eq(10000, 50000, 7)
    if len(sys.argv) == 2:
        production_sweeps = int(sys.argv[1])
        logging.info("simulating with")
        sim.run_eq(10000, production_sweeps, 7)

