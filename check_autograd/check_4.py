import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
import matplotlib.pyplot as plt


def tanh(x):  # Define a function
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)


grad_tanh = grad(tanh)

print(grad_tanh(1.0))
print((tanh(1.0001) - tanh(0.9999)) / 0.0002)


x = np.linspace(-3, 3, 200)
plt.plot(x, tanh(x),
         x, egrad(tanh)(x),  # first  derivative
         x, egrad(egrad(tanh))(x),  # second derivative
         x, egrad(egrad(egrad(tanh)))(x),  # third  derivative
         x, egrad(egrad(egrad(egrad(tanh))))(x),  # fourth derivative
         x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x),  # fifth  derivative
         x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x))  # sixth  derivative
plt.show()
