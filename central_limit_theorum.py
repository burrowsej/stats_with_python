import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, skewnorm


# not a normal distribution
def get_data(n):
    data = np.concatenate((expon.rvs(scale=1, size=n//2),
                           skewnorm.rvs(5, loc=3, size=n//2)))
    np.random.shuffle(data)
    return data
plt.hist(get_data(2000), bins='auto')

d10 = get_data(10)
print(d10.mean())

# 1000 samples of 10 points, mean of each
means = [get_data(10).mean() for i in range(1000)]
plt.hist(means, bins='auto')
print(np.std(means))

# 1000 samples of 100 points, mean of each
means = [get_data(100).mean() for i in range(1000)]
plt.hist(means, bins='auto')
print(np.std(means))

# 1000 samples of 100 points, mean of each
means = [get_data(100).mean() for i in range(1000)]
plt.hist(means, bins='auto')
print(np.std(means))
# spread of means decreases with larger samples

num_samps = [10, 50, 100, 500, 1000, 5000, 10000]
stds = []
for n in num_samps:
    stds.append(np.std([get_data(n).mean() for i in range(1000)]))
plt.plot(num_samps, stds, 'o', label="Obs scatter")
plt.plot(num_samps, 1 / np.sqrt(num_samps), label="Rando function", alpha=0.5)
plt.legend()
# std is related to the inverse sqrt of number of samples

# distribution of sample means approaches a normal distribution
# width of distribution depends on number of samples that are used to produce each mean

# if you have N sampes, the mean of your samples is distributed as per a normal around the true mean
# with std sigma/sqrt(N)
# another way of saying this is that if you go from N1 data points to N2 data points, you can determine
# the mean sqrt(N1/N2) more accurately. 4 times as many samples does not give 4 times more accuracy, only double.


n = 1000
data = get_data(n)
sample_mean = np.mean(data)
uncert_mean = np.std(data) / np.sqrt(n)
print(f"We have determined the mean of the population to be {sample_mean:.2f} +- {uncert_mean:.2f}")

