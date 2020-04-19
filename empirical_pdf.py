import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
          5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
ys = [0.2, 0.165, 0.167, 0.166, 0.154, 0.134, 0.117,
      0.108, 0.092, 0.06, 0.031, 0.028, 0.048, 0.077,
      0.103, 0.119, 0.119, 0.103, 0.074, 0.038, 0.003]

plt.scatter(xs, ys)
plt.xlabel("x")
plt.ylabel("Observed PDF")

# continuous distribution where we've binned the data

x = np.linspace(min(xs), max(xs), 1000)

# splines would be faster but less control
# returns a function that can be fed new xs to get ys
# linear interpolation is fast but inaccurate with sparse & volatile data
y1 = interp1d(xs, ys)(x)
# nearset is not very useful for continuous
y2 = interp1d(xs, ys, kind="nearest")(x)
# quadratic best for quickly changing gradiants
y3 = interp1d(xs, ys, kind="quadratic")(x)
# cubic may fit better but slower - but if dataa is that veriable,
# interpolation may not be the way forward - may introduce bias
y4 = interp1d(xs, ys, kind="cubic")(x)

# alternative - use splines!
from scipy.interpolate import splev, splrep
y5 = splev(x, splrep(xs, ys))

plt.scatter(xs, ys, s=30, label="Data")
plt.plot(x, y1, label="Linear (default)")
plt.plot(x, y2, label="Nearest", alpha = 0.3)
plt.plot(x, y3, label="Quadratic", ls="-")
plt.plot(x, y4, label="Cubic", ls="-")
plt.plot(x, y5, label="Spline", ls="--", alpha=0.5)
plt.legend()

# use linear as default, then quad

# to get pdf/cdf/sf we need to integrate
# scipy.integrate.trapz - low accuracy, high speed, accuracy scales as O(h)
# (h is distance between samples), trapz is effectively linear interpolation
# scipy.integrate.simps - medium accuracy, pretty high speed, accuracy scales as O(h^2)
# simpsons is effectively quadratic interpolation
# scipy.integrate.quad - high accuracy, low speed, accuracy arbitrary
# quad takes a function, not xs and ys, words to converge unitl it is satisfied
# could take the interp1d as the function

# simpsons is best of both worlds
from scipy.integrate import simps

def get_prob(xs, ys, a, b, resolution=1000):
    ''' a and b are bounds '''
    # need to normalise to ensure area under curve is 1
    x_norm = np.linspace(min(xs), max(xs), resolution)
    y_norm = interp1d(xs, ys, kind="quadratic")(x_norm)
    normalisation = simps(y_norm, x=x_norm)
    x_vals = np.linspace(a, b, resolution)
    y_vals = interp1d(xs, ys, kind="quadratic")(x_vals)
    return simps(y_vals, x=x_vals) / normalisation

def get_cdf(xs, ys, v):
    return get_prob(xs, ys, min(xs), v)

def get_sf(xs, ys, v):
    return 1 - get_cdf(xs, ys, v)

# slightly greater than 1 - not entirely accurate!
print(get_prob(xs, ys, 0, 10))

plt.cla()
v1, v2 = 6, 9.3
area = get_prob(xs, ys, v1, v2)

plt.scatter(xs, ys, s=30, label="Data")
plt.plot(x, y3, linestyle="-", label="Interpolation")
plt.fill_between(x, 0, y3, where=(x>=v1)&(x<=v2), alpha=0.2)
plt.annotate(f"p = {area:.3f}", (7, 0.05))
plt.legend()

x_new = np.linspace(min(xs), max(xs), 100)
cdf_new = [get_cdf(xs, ys, i) for i in x_new]
cheap_cdf = y3.cumsum() / sum(y3)

plt.cla()
plt.plot(x_new, cdf_new, label="Interpolated CDF")
plt.plot(x, cheap_cdf, label="Super cheap CDF for specific cases", ls="--")
plt.ylabel("CDF")
plt.xlabel("x")

