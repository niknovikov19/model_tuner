import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos, sqrt, arctan, tan
import xarray as xr


def F_lin(Ru, Rc):
    a = 0.8
    b = -1
    m = 0.3
    Rc_hat = a * Rc - b * Ru + m
    return Rc_hat

phi0 = 0.45 * pi

def F_pol(u, c):
    m = 10
    r0 = c / sin(phi0)
    s = sqrt(u**2 + c**2)
    r = r0 - m * (phi0 - arctan(c / u))
    c_hat = (u * (m * c + u * r) - r**2 * s) / (m * u - c * r)
    return c_hat

F = F_pol

#pfr = np.array([0.25, 0.5, 1., 1.5])
pfr = np.array([1.5, 2.])
Rc0 = pfr

alpha = 0.0001
n_iter = 10000

# Initial guess (K, 1)
Ru = Rc0.copy()
Rc = Rc0.copy()

# True solution
#Ru_sol = ((a - 1) * Rc0[1] * pfr + m) / b
Ru_sol = Rc0 / tan(phi0)
#Ru = Ru_sol.copy() * 1.2
Ru = Rc0.copy() / tan(phi0 - 0.02 * pi)

#Ru[:, 1] /= 50
#Rc[:, 1] *= 50

Ru_data = np.zeros((len(pfr), n_iter))
Rc_data = np.zeros((len(pfr), n_iter))
Ru_data[:, 0] = Ru
Rc_data[:, 0] = Rc

for n in range(1, n_iter):
    # Regression
    b, a = np.polyfit(Rc.ravel(), Ru.ravel(), 1)
    
    # Choose Ru_hat
    Ru_hat = a + b * Rc0

    # "Simulation"
    Rc_hat = F(Ru_hat, Rc0)
    
    # Step
    Ru += alpha * (Ru_hat - Ru)
    Rc += alpha * (Rc_hat - Rc)

    Ru_data[:, n] = Ru.copy()
    Rc_data[:, n] = Rc.copy()

plt.figure()
for n in range(0, n_iter, 100):
    plt.plot(Ru_data[:, n], Rc_data[:, n], '.-')
plt.plot(Ru_data[:, -1], Rc_data[:, -1], 'k', linewidth=2)
plt.plot(Ru_sol, Rc0, 'r--')
plt.plot((Ru_data[:].min(), Ru_data[:].max()),
         (Rc0.max(), Rc0.max()), 'k--')
plt.xlabel('Ru')
plt.ylabel('Rc')
#plt.xlim(0, 1)
#plt.ylim(0, 2)
plt.show()