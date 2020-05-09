import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("loaded_500.txt")

# is Tommy rolling too many 6s?

unique, counts = np.unique(data, return_counts=True)
print(unique, counts)
plt.hist(data, bins=6)

num_sixes = (data == 6).sum()
num_total = data.size

# what is the chance we roll 98 or more 6s with a fair die?
from scipy.stats import binom
n = np.arange(num_total)
prob_n = binom.pmf(n, num_total, 1/6)
plt.clf()
plt.plot(n, prob_n, label="Prob num")
plt.axvline(num_total / 6, ls="--", lw=1, color='orange', label="Mean num")
plt.axvline(num_sixes, ls=":", color='black', label="Obs num")
plt.xlabel(f"Num sizes rolled out of {num_total} rolls")
plt.ylabel("Probability")
plt.legend()

plt.clf()
d = binom(num_total, 1/6) # pre-configures the distribution
plt.plot(n, d.sf(n)) # survival function
sf = d.sf(num_sixes)
plt.axvline(num_sixes, ls="--", lw=1, color='black')
plt.axhline(sf, ls="--", lw=1, color='black')
plt.xlabel(f"Num sizes")
plt.ylabel("SF")
print(f"Only {sf * 100:.1f}% of the time with a fair dice would you roll this many or more sizes")

# needed to have picked a p value