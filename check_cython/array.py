import time
import numpy
from stdlib.tprint import tprint

total = 0
arr = numpy.arange(1000000000)

t1 = time.time()

tprint("start")
for k in arr:
    total = total + k
tprint("Total = ", total)

t2 = time.time()
t = t2 - t1
print("%.20f" % t)
