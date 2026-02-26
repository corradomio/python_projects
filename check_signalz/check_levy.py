import signalz
import matplotlib.pyplot as plt

x = signalz.levy_noise(1000, alpha=1.5, beta=0.5, sigma=1., position=-2)

plt.hist(x, bins=100)
plt.show()

