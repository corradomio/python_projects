import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# --- 1. Define the model parameters for simulation ---
# For an ARCH(1) model:
# omega (constant variance component)
# alpha (coefficient for the lagged squared residual, q=1)
# beta (coefficient for lagged variance, beta=0 for pure ARCH)
n_samples = 250
omega_param = 0.5
alpha_param = 0.4
beta_param = 0.0 # Set beta to 0 for an ARCH(1)

# --- 2. Simulate the series ---
# The arch_model can be used to set up a simulation
# We create a dummy model configuration and then use its simulate method
# The 'rescale=False' is used to get the raw simulated values
sim_model = arch_model(None, mean='Constant', vol='ARCH', p=1, q=0)
sim_data = sim_model.simulate([omega_param, alpha_param], n_samples, burn_in=50)

# The simulate method returns a Series with 'data' (simulated residuals) and 'variance'
simulated_returns = sim_data.data
simulated_variance = sim_data.variance

# --- 3. Visualize the simulated time series and its volatility ---
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

axes[0].plot(simulated_returns)
axes[0].set_title('Simulated ARCH(1) Returns')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Returns')

axes[1].plot(simulated_variance, color='red')
axes[1].set_title('Simulated ARCH(1) Conditional Variance')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Variance')

plt.tight_layout()
plt.show()
