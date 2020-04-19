import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# every plot should have a quantification

data = np.loadtxt("dataset.txt")
plt.hist(data, bins=50)

# CENTRALITY
# mean

mean = np.mean(data)
print(mean, data.mean(), np.average(data))
# np.average is mean except can optionally weight it

# median
# median less sensitive to outliers and/or bad data
np.median(data)
np.quantile(data, 0.5)

# mode
st.mode(data) # lucy we got only one, that is because only 3dp, need to bin
# bin data for continuous data first
hist, edges = np.histogram(data, bins=100)
edge_centres = 0.5 * (edges[1:] + edges[:-1])
mode = edge_centres[hist.argmax()]
print(mode) # dependant on bin size


# gaussian kde better for mode on continuous - no binning
kde = st.gaussian_kde(data)
xvals = np.linspace(data.min(), data.max(), 1000)
yvals = kde(xvals)
mode = xvals[yvals.argmax()]
print(mode)
plt.hist(data, bins=100, density=True, label="Data hist", histtype="step")
plt.plot(xvals, yvals, label="KDE")
plt.axvline(mode, label="Mode", color="g")
plt.legend()

# WIDTH & BALANCE

# variance
# sample var, divide by n or n-1? sample N-1, pop N
# biased (n) or unbiased (n-1) estimate of polulation variance
# Bessel's correction
variance = np.var([1,2,3,4,5])
print(variance)	
variance = np.var([1,2,3,4,5], ddof=1)
print(variance)

# standard deviation
# sqrt of variance
std = np.std(data)
print(std, std**2)

# guassian/normal approximation
xs = np.linspace(data.min(), data.max(), 100)
ys = st.norm.pdf(xs, loc=mean, scale=std)

# skewness
# measure of asymetry- 3rd moment. 2nd moment = variance, 1st = 0
skewness = st.skew(data)
print(skewness)
ps = st.skewnorm.fit(data)
ys2 = st.skewnorm.pdf(xs, *ps)
plt.hist(data, bins=50, density=True, histtype="step", label="Data")
plt.plot(xs, ys, label="Normal approximation")
plt.plot(xs, ys2, label="Skewormal approximation")
plt.legend()
plt.ylabel("Probability")

# kurtosis
# similar to skewness but power of 4 instead of 3
# less a measure of asymmetry
kurtosis = st.kurtosis(data, fisher=False)
print(kurtosis, st.kurtosis(data))
# fisher normalises. Unnormalised, guussian=3, so take 3 away

# quantiles
# trick to sample percentiles using norm and 0, 100
# otherwise linspace is missing info at tails
ps = 100 * st.norm.cdf(np.linspace(-3, 3, 30))
ps = np.insert(ps, 0, 0)
ps = np.insert(ps, -1, 100)
x_p = np.percentile(data, ps)

xs = np.sort(data)
ys = np.linspace(0, 1, len(data))
plt.plot(xs, ys * 100, label="ECDF")
plt.plot(x_p, ps, label="Percentiles", marker=".", ms=10)
plt.legend()
plt.ylabel("Percentile")
# a way to reduce a distribution down to x points - great!

