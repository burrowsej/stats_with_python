import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_original = pd.read_csv("Diabetes.csv")

# turn zeros in certain columns into NaN
cols = [c for c in df_original.columns if c not in ['Pregnancies', 'Outcome']]
df = df_original.copy()
df[cols] = df[cols].replace({0: np.NAN})

df.info()
df.describe()

# scatter matrix/plot
pd.plotting.scatter_matrix(df, figsize=(7,7))

# clarifying the binary outputs
df2 = df.dropna()
colours = df2["Outcome"].map(lambda x: "#44d9ff" if x else "#f95b4a")
pd.plotting.scatter_matrix(df2, figsize=(7,7), color=colours)

# correlation
df.corr() # corrlation table

sns.heatmap(df.corr()) # put it in a heatmap to clarify

sns.heatmap(df.corr(), annot=True, cmap="viridis", fmt="0.2f")
# great for quick insites on new data

# 2D histograms
df2 = pd.read_csv("height_weight.csv")
df2.info()
df2.describe()

plt.hist2d(df2["height"], df2["weight"], bins=20, cmap="magma")
plt.xlabel("Height")
plt.ylabel("Weight")

# contour plot
hist, x_edge, y_edge = np.histogram2d(df2["height"], df2["weight"], bins=20)
x_center = 0.5 * (x_edge[1:] + x_edge[:-1])
y_center = 0.5 * (y_edge[1:] + y_edge[:-1])

plt.contour(x_center, y_center, hist, levels=10)
plt.xlabel("Height")
plt.ylabel("Weight")
# doesn't look great - not smooth - not enough data
# would want 14,000 plus data points

# kde plot
sns.kdeplot(df2["height"], df2["weight"], cmap="viridis", bw=(2,20))
plt.hist2d(df2["height"], df2["weight"], bins=20, cmap="magma", alpha=0.3)
# kernal smooths the data by a specified amount - gaussian kernal
# can use bw as number or specific tuple for each axis

# easy, good for presentations
sns.kdeplot(df2["height"], df2["weight"], cmap="viridis", shade=True)

# simple scatter
m = df2["sex"] == 1
plt.scatter(df2.loc[m, "height"], df2.loc[m, "weight"], c="#16c6f7", s=1, label="Male")
plt.scatter(df2.loc[~m, "height"], df2.loc[~m, "weight"], c="#ff8b87", s=1, label="Female")
plt.legend(loc=2)
plt.xlabel("Height")
plt.ylabel("Weight")

# treating points with probability
# MCMC chains and posterior samples
# instructor produced this lib

params = ["height", "weight"]
male = df2.loc[m, params].values
female = df2.loc[~m, params].values

from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(male, parameters=params, name="Male", kde=1.0, color="b")
c.add_chain(female, parameters=params, name="Female", kde=1.0, color="r")
c.configure(contour_labels='confidence', usetex=False, serif=False)
c.plotter.plot(figsize=2.0)
sns.despine(left=True, bottom=True)
# shows 68% and 95% confidence intervals
# good for hypothesis testing
# does a data point come from the distribution?
# probability surfaces

# instead of looking at contours we can look at 1d distributions
c.plotter.plot_summary(figsize=2.0)
sns.despine(left=True, bottom=True)
# similar to violin plot

