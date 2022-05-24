
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
import pandas as pd
import seaborn as sns
from palettable.colorbrewer.sequential import Reds_3
from scipy import linalg

mycmap = mpl.colors.LinearSegmentedColormap.from_list('custom',
                                             [(0,    'white'),
                                              (0.3,  'white'),
                                              (1,    '#ff0000')], N=126)

mybwr = mpl.colors.LinearSegmentedColormap.from_list('custom',
                                             [(0,    'blue'),
                                              (0.2,  'blue'),
                                              (0.3,  'white'),
                                              (0.6,  'white'),
                                              (1,    '#ff0000')], N=126)

import sys
path = sys.argv[1]
outpath = sys.argv[2]
vmaxp = sys.argv[3]

contact = np.loadtxt(path)
contact /= max(np.array(contact).diagonal())
plt.figure(figsize=(12,10))
#plt.imshow(contact, cmap=mycmap, vmin = 0, vmax = contact.mean()+vmaxp*contact.std())
plt.imshow(contact, cmap=mycmap, vmin = 0, vmax = vmaxp)
plt.colorbar()
plt.title("max : {:.4f}, mean {:.4f}".format(np.max(contact), contact.mean()))
plt.savefig(outpath + "/contact.png")

