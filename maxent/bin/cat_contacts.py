import os
import sys

import numpy as np

print(sys.argv)


files = sys.argv[1:-1]
output_file = sys.argv[-1]

rows, cols = np.shape(np.loadtxt(files[0]))
output = np.zeros((rows,cols))

for file in files:
    output += np.loadtxt(file)
    os.remove(file)

np.savetxt(output_file, output, fmt="%d", delimiter=" ")
