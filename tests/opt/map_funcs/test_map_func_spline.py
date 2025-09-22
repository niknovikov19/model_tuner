import matplotlib.pyplot as plt
import numpy as np

from model_tuner.opt.map_funcs import MapFunc1DSpline


np.random.seed(0)
n = 6
xx = np.linspace(0, 3, n)
yy = np.sinh(xx) + 0.1 * np.random.randn(n)

map_func = MapFunc1DSpline(x_limits=(0, np.inf),
                           y_limits=(0, np.inf))
map_func.fit(xx, yy)

x_ = np.linspace(-1, 4, 100)
y_ = map_func.f(x_)
x_inv = map_func.f_inv(y_)

plt.figure()
plt.plot(xx, yy, 'o', label='data')
plt.plot(x_, y_, '-', label='spline')
plt.plot(x_inv, y_, '--', label='spline inv')
plt.legend()
plt.show()