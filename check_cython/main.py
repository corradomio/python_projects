import pyximport
pyximport.install()

import fib
import array


print(fib.__file__)

print(fib.fib(10))
