import numpy as np
import pickle
import pandas as pd
filename = "load.csv"

# manual
cols = None
data = []
with open(filename) as f:
    for line in f.readlines():
        vals = line.replace("\n", "").split(",")
        print(vals)
        if cols is None:
            cols = vals
        else:
            data.append([float(x) for x in vals])

d0 = pd.DataFrame(data, columns = cols)
print(d0.dtypes)
d0.head()

# numpy loadtxt
d1 = np.loadtxt(filename, skiprows = 1, delimiter= ",")
print(d1.dtype)
print(d1[:5, :])

# numpy genfromtxt
d2 = np.genfromtxt(filename, delimiter=",", names=True, dtype=None)
print(d2.dtype)
print(d2['A'][:5])

# pandas
d3 = pd.read_csv('load.csv')
print(d3.dtypes)
d3.plot()

# from a pickle
with open("load_pickle.pickle", "rb") as f:
    d4 = pickle.load(f)
    