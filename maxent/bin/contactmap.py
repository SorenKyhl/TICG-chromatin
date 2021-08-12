import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from scipy import linalg
import seaborn as sns
mpl.use('Agg')

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

df = pd.read_csv(it_dir + "/data_out/contacts.txt", delimiter=" ", header=None)
df = df.fillna(0)
mean1 = df.stack().mean()

plt.figure(figsize=(12,10))
sns.heatmap(df/(mean1), vmax=1, cmap=mycmap);
plt.savefig(it_dir + "/contact" + str(it) + ".png")
