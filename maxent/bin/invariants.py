import sys

import numpy as np

N = float(sys.argv[1])
b = float(sys.argv[2])
v = float(sys.argv[3])


def R(N,b,v):
    return b*np.sqrt(N)

def sqrtNbar(N,b,v):
    return b**3 * np.sqrt(N) / v

print(sqrtNbar(N,b,b))
print(R(N,b,v)**3/(v*N))
