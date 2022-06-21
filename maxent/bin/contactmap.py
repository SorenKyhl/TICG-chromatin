import sys

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
it = int(sys.argv[1])
it_dir = "iteration" + str(it)

df = pd.read_csv(it_dir + "/production_out/contacts.txt", delimiter=" ", header=None)
df = df.fillna(0)
df /= np.max(df)

mean1 = df.stack().mean()
std1 = df.stack().std()
plt.figure(figsize=(12,10))
plt.imshow(df, cmap=mycmap, vmin = 0, vmax = mean1+0.1*std1)
plt.colorbar()
plt.savefig(it_dir + "/contact" + str(it) + ".png")

#sns.heatmap(df/(1.5*mean1), vmax=1, cmap=Reds_3.mpl_colormap)

# red-WHITE:
#sns.heatmap(df/(mean1), vmax=1, cmap=mycmap)
#plt.savefig(it_dir + "/contact" + str(it) + ".png")

#R-B
#sns.heatmap(df/(mean1), vmax=1, cmap=mybwr)
#plt.savefig(it_dir + "/contact" + str(it) + "bwr.png")
