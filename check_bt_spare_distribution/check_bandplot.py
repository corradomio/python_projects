import matplotlib.pyplot as plt
from matplotlibx.patches import rectangle
import matplotlibx.pyplot as pltx


# rectangle(xy=[.1, .1, .9, .5])
# plt.show()

pltx.bandplot([1,2,3,4,5], yerr=1, alpha=0.5)
plt.show()


