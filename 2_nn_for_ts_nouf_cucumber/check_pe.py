import matplotlib.pyplot as plt
import torchx.nn as nnx


pe = nnx.positional_encoding(256, 768, astensor=False)

plt.figure(figsize=(12, 6), dpi=150)
plt.imshow(pe)
plt.tight_layout()

plt.savefig('positional_encoding.jpg', dpi=300)
# plt.show()
