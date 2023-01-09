import numpy as np
import matplotlib.pyplot as plt

plt.hist(np.random.normal(0, 0.05, 10**3), bins=100, ec="black")
plt.show()