import matplotlib
import matplotlib.pyplot as plt

x = 1
y = 2

z = x + y
print(z)

matplotlib.use('qt5agg')

plt.figure()
plt.plot([1, 2, 3], [1, 2, 3])
plt.show()