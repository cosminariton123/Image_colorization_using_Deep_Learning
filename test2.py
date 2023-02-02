import numpy as np
import matplotlib.pyplot as plt


a = np.random.uniform(-1, 1, 10**5)
a = ((a+1) / 2 ) * 255

a = np.random.uniform(0, 255, 10**5)
a = a / 255 * 2 - 1


plt.hist(a, bins=200, ec="black")
plt.show()