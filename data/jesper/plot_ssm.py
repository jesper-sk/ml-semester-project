import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from scipy.spatial.distance import pdist, squareform

METRIC = 'euclidean'

f = np.genfromtxt('../F.txt')

min_key_nr = np.min(f[np.nonzero(f)])
f[f==0] = np.nan
fn = np.maximum(f - min_key_nr, 0) % 12

ds = [squareform(pdist(fn[:,x], metric=METRIC)) for x in 
      product(range(4), range(4))]
for d in ds:
    d -= np.nanmin(d)
    d /= (np.nanmax(d) - np.nanmin(d))

for x in product(range(4), range(4)):
    fig, ax = plt.subplots(figsize=(20,20))
    ax.matshow(ds[x[0]*4 + x[1]], interpolation='nearest')
    ax.grid(True)
    plt.savefig("../ssm-between-voices/%sv%s.png" % x)
