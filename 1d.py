import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d1 = np.loadtxt("example_1.txt")
d2 = np.loadtxt("example_2.txt")
print(d1.shape, d2.shape)

# historam
plt.hist(d1, label="D1", alpha=0.5)
plt.hist(d2, label="D2", alpha=0.5)
plt.legend()
plt.ylabel("Counts")

bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()))
plt.hist(d1, bins=bins, label="D1", alpha=0.5)
plt.hist(d2, bins=bins, label="D2", alpha=0.5)
plt.legend()
plt.ylabel("Counts")
plt.plot()

bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()))
plt.hist(d1, bins=bins, label="D1", density = True, histtype="step", ls=":")
plt.hist(d2, bins=bins, label="D2", alpha=0.5, density = True, histtype="step", lw=3)
plt.legend()
plt.ylabel("Probability")
plt.plot()

bins = 50
plt.hist([d1, d2], bins=bins, label="Stacked", density = True, histtype="barstacked", alpha=0.5)
plt.hist(d1, bins=bins, label="D1", density = True, histtype="step", ls=":")
plt.hist(d2, bins=bins, label="D2", alpha=0.5, density = True, histtype="step", lw=3)
plt.legend()
plt.ylabel("Probability")
plt.plot()

# bee swarms
df = pd.DataFrame({
    "value": np.concatenate((d1, d2)),
    "type": np.concatenate((np.ones(d1.shape), np.zeros(d2.shape)))})
df.info()

sns.swarmplot(df["value"])

sns.swarmplot(x="type", y="value", data=df, size=2.5)
# good for categorical data, e.g. swarm for each month or day of week

# box plots
sns.boxplot(x="type", y="value", data = df, whis=2.0)
sns.swarmplot(x="type", y="value", data=df, size=2, color="k", alpha=0.3)
sns.despine(trim = True)
# whisker lines default to 1.5x IQR
# big loss of info with boxplot - really just 5 numbers
# mainly for comparing a lot of distributions
# or include the swarm or violin

# violins - better than box
sns.violinplot(x="type", y="value", data = df)
sns.swarmplot(x="type", y="value", data=df, size=2, color="k", alpha=0.3)

sns.violinplot(x="type", y="value", data = df, inner="quartile", bw=0.2)
# better to undersmooth than oversmooth

# Empirical Cumulative Distribution Functions ECDF
sd1 = np.sort(d1)
sd2 = np.sort(d2)
cdf = np.linspace(1/d1.size, 1, d1.size)
plt.plot(sd1, cdf, label="d1 CDF")
plt.plot(sd2, cdf, label="d2 CDF")
plt.hist(d1, histtype="step", density=True, alpha=0.3)
plt.hist(d2, histtype="step", density=True, alpha=0.3)
# not that useful for visual, but can pass to to tests etc.

# describe
df = pd.DataFrame({"Data1": d1,
                   "Data2": d2})
df.describe()
# best for starting analysis