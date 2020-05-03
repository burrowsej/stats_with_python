from scipy.stats import norm, uniform
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt

# rvs= random value sample
plt.hist(norm.rvs(loc=10, scale=2, size=1000), bins='auto')

# rolling 3 normal dice
samples = np.ceil(uniform.rvs(loc=0, scale=6, size=(100000, 3))).sum(axis=1)
plt.clf()
plt.hist(samples, bins='auto')

# rejection sampling
# function - p(x)=sin(x^2)+1 from 0->4
# not a pdf, area under is not 1, but doesn't need normalising here
def pdf(x):
    return np.sin(x**2) + 1

xs = np.linspace(0, 4, 200)
ps = pdf(xs)
plt.clf()

n = 300
random_x = uniform.rvs(loc=0, scale=4, size=n)
random_y = uniform.rvs(loc=0, scale=2, size=n)

passed = random_y <= pdf(random_x)
plt.scatter(random_x[passed], random_y[passed])
plt.scatter(random_x[~passed], random_y[~passed], marker="x", s=30,alpha=0.5)
plt.plot(xs, ps, c="w")
plt.fill_between(xs, 0, ps, color='black', alpha=0.1)
plt.xlim(0, 4), plt.ylim(0,2)
# about half the sampling is rejected - inefficient

n2 = 100000
x_test = uniform.rvs(scale=4, size=n2)
x_final = x_test[uniform.rvs(scale=2, size=n2) <= pdf(x_test)]
plt.clf()
plt.hist(x_final, density=True, histtype="step", label="Sampled dist", bins='auto')
plt.plot(xs, ps / simps(ps, x=xs), c="black", label="Empirical PDF")
plt.legend(loc=2)

# inversion sampling
# pdf: p(x) = 3x**2 from 0->1 (normalised)
# cdf = x**3, invert: x = y**3, so y = x**(1/3)
# so x = CDF**(1/3)
def pdf(x):
    return 3 * x**2
def cdf(x):
    return x**3
def icdf(cdf):
    return cdf**(1/3)
xs = np.linspace(0, 1, 100)
pdfs = pdf(xs)
cdfs = cdf(xs)
n = 2000
u_samps = uniform.rvs(size=n)
x_samps = icdf(u_samps)
f, axs = plt.subplots(ncols=2, figsize=(10,4))
axs[0].plot(xs, pdfs, color="black", label='PDF')
axs[0].hist(x_samps, density=True, histtype='step', label="Sampled dist", bins='auto')
axs[1].plot(xs, cdfs, color="black", label='CDF')
axs[1].hlines(u_samps, 0, x_samps, linewidth=0.1, alpha=0.3)
axs[1].vlines(x_samps, 0, u_samps, linewidth=0.1, alpha=0.3)
axs[0].legend(), axs[1].legend()

# now using on the sin function from before
from scipy.interpolate import interp1d
def pdf(x):
    return np.sin(x**2) + 1
xs = np.linspace(0, 4, 10000)
pdfs = pdf(xs)
cdfs = pdfs.cumsum() / pdfs.sum() # dangerous - will never actually be 0
u_samps = uniform.rvs(size=4000)
x_samps = interp1d(cdfs, xs)(u_samps) # if one is less than 1/10000 will throw error
f, axs = plt.subplots(ncols=2, figsize=(10,4))
axs[0].plot(xs, pdfs/4.747, color="black", label='PDF')
axs[0].hist(x_samps, density=True, histtype='step', label="Sampled dist", bins='auto')
axs[1].plot(xs, cdfs, color="black", label='CDF')
axs[1].hlines(u_samps, 0, x_samps, linewidth=0.1, alpha=0.3)
axs[1].vlines(x_samps, 0, u_samps, linewidth=0.1, alpha=0.3)
axs[0].legend(), axs[1].legend()

