from random import random
import numpy as np
import autograd.numpy as anp
from autograd import grad

lx = [random() for i in range(10)]
lm = [[random() for i in range(10)] for j in range(10)]


M = anp.array(lm, dtype=float)
x0 = anp.array([i % 2 for i in range(10)], dtype=float)
y0 = np.dot(M, x0)


x = anp.array(lx)

def loss(x):
    return anp.linalg.norm(anp.dot(M, x) - y0)


grad_loss = grad(loss)

eprev = 200
ecurr = 100
while ecurr < eprev:
    eprev = ecurr
    y = np.dot(M, x)
    ecurr = anp.linalg.norm(y-y0)
    print(ecurr)
    dl = grad_loss(x)
    x -= 0.0005*dl


y = np.dot(M, x)
print(x)
pass
