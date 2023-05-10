import matplotlib.pyplot as plt
import sys

from pylib.utils import epilib


def plot_contactmap(sim_dir):
    sim = epilib.Sim(sim_dir)
    sim.plot_contactmap()
    plt.savefig("contactmap.png")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # impute sim_dir
        try:
            plot_contactmap("production_out")
        except:
            print("usage: plot_contactmap simulation_dir")
    else:
        plot_contactmap(sys.argv[1])
