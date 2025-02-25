import matplotlib.pyplot as plt


plt.ion()
plt.figure()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.draw()

mng = plt.get_current_fig_manager()
mng.window.showMaximized()  # Maximize the figure window

plt.show()

print('Finished')

input("Press Enter to exit...") 