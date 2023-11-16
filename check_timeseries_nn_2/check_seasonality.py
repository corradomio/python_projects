import numpy as np
import numpyx as npx
import matplotlib.pyplot as plt

# 30 days 12 months 10 years
n = 30*12*10
x = np.arange(n)
y = 1. + .001*x \
    + 0.01*np.sin(2*np.pi*7/n*x) \
    # + 0.05*np.sin(2*np.pi*30/n*x) \
    # + 0.03*np.sin(2*np.pi*360/n*x) \

plt.plot(x, y)
plt.show()

f = np.fft.rfft(y)
m = npx.chop(np.abs(f))
p = npx.chop(np.angle(f))

z = np.fft.irfft(f)

plt.plot(x, z)
plt.show()

fre, fim = npx.chop(npx.to_rect(f))
fro, fph = npx.chop(npx.to_polar(f))

f1 = npx.chop(npx.from_rect(fre, fim))
f2 = npx.chop(npx.from_polar(fro, fph))

pass