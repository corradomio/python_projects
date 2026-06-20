import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)            # Scale 1: Range [-1, 1]
y2 = np.exp(x)            # Scale 2: Range [1, 22026]

# Create the figure and primary axis
fig, ax1 = plt.subplots()

# Plot the first dataset on the left Y-axis
color = 'tab:blue'
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Sine Wave (Left)', color=color)
ax1.plot(x, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a twin axis sharing the same X-axis
ax2 = ax1.twinx()

# Plot the second dataset on the right Y-axis
color = 'tab:red'
ax2.set_ylabel('Exponential (Right)', color=color)
ax2.plot(x, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Prevent layout overlapping
fig.tight_layout()
plt.show()
