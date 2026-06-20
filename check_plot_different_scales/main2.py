import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)            # Scale 1: Range [-1, 1]
y2 = np.exp(x)            # Scale 2: Range [1, 22026]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left Subplot: Standard single-scale plot
axes[0].plot(x, y1, color='blue')
axes[0].set_title("Standard Subplot")

# Right Subplot: Twin Y-axis scales
axes[1].plot(x, y1, color='blue')
axes[1].set_ylabel("Left Scale", color='blue')

# Generate the twin scale exclusively on the second subplot grid element
ax_twin = axes[1].twinx()
ax_twin.plot(x, y2, color='red')
ax_twin.set_ylabel("Right Scale", color='red')
axes[1].set_title("Subplot with Dual Scales")

fig.tight_layout()
plt.show()
