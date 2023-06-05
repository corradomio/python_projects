import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi

a1 = .2
b1 = .03
f = 2 * np.pi / 12.
a2 = .1
b2 = .1

t = np.arange(0, 11*12, 1/12, dtype=float)
y = (a1 + b1*t)*np.sin(f*t) + (a2 + b2*t)

plt.plot(t, y)
plt.show()
