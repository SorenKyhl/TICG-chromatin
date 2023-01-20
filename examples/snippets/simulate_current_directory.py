from pylib.pysim import Pysim

if __name__ == "__main__":
    sim = Pysim.from_directory(".")
    sim.run_eq(10000, 50000, 7)
