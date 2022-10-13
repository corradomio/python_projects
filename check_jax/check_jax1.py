import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import device_put

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
y = jnp.dot(x, x.T).block_until_ready()
print(y)

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
y = jnp.dot(x, x.T).block_until_ready()
print(y)


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


x = random.normal(key, (1000000,))
y = selu(x).block_until_ready()
print(y)
