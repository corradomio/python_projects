# Source - https://stackoverflow.com/q/64621237
# Posted by Elena Greg, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-18, License - CC BY-SA 4.0

import matplotlib.pyplot as plt

x_values1=[1,2,3,4,5]
y_values1=[12,21,42,54,-1]

x_values2=[0.1,0.2,0.3,0.4,0.5]
y_values2=[5000,3000,4000,1000,2000]

x_values3=[150,200,250,300,350]
y_values3=[30,20,50,40,10]

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(10,6))

ax1.plot(x_values1, y_values1)

parax6 = ax6.twinx().twiny()

ax6.plot(x_values2, y_values2, c="r", label="red")
parax6.plot(x_values3, y_values3, c="b", label="blue")
ax6.legend(loc="lower left")
parax6.legend(loc="upper right")

parax6.set_xticks([])
parax6.set_yticks([])
for side in ['top','right','left','bottom']:
    parax6.spines[side].set_visible(False)

ax6.set_xticks([])
ax6.set_yticks([])
for side in ['top','right']:
    ax6.spines[side].set_visible(False)


plt.tight_layout()
plt.show()
