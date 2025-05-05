from matplotlib import pyplot as plt
import numpy as np

from model_tuner.opt.map_funcs import MapFunc1DSigmoidLine
from model_tuner.opt.map_funcs import MapFitParams


# Create some sample data points
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.5, 3.5, 4.2, 5.1, 6.0])

# Initialize the MapFunc1DRational21 object
map_func = MapFunc1DSigmoidLine(
    x_limits=(-np.inf, np.inf),
    #y_limits=(-np.inf, np.inf),
    y_limits=(0, np.inf)
)

map_fit_params = MapFitParams(
    xtol=None,
    ftol=0.01,
    max_nfev=2000,
    method='trf',
    return_first_guess=0
)

def f(x):
    y = 2 + 5 / (1 + np.exp(-2 * (x + 3))) ** 2 + 0.5 * x
    #return y
    return np.maximum(y, 0)

x = np.linspace(-6, 2, 200)
y = f(x)

xx = np.linspace(-6, 2, 5)
yy = f(xx)

# Fit the function to the data points
map_func.fit(xx, yy, map_fit_params)
print(map_func.par)

# Evaluate the fitted function at the data points
yhat = map_func.apply(x)

plt.figure()
plt.plot(x, y)
plt.plot(x, yhat, 'r')
plt.plot(xx, yy, 'k.')
plt.show()