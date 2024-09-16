from pprint import pprint
from pygadx import Solution
import numpy as np

# ---------------------------------------------------------------------------
sol = Solution(bool, 10)

print(sol.num_genes)
print(sol.gene_type)
print(sol.gene_space)

pop = sol.population(5)
pprint(pop)

# ---------------------------------------------------------------------------
sol = Solution(np.int8, num_genes=10, gene_space=(0, 10))

print(sol.num_genes)
print(sol.gene_type)
print(sol.gene_space)

pop = sol.population(5)
pprint(pop)

# ---------------------------------------------------------------------------
sol = Solution(float, num_genes=10, gene_space=(-1, 1))

print(sol.num_genes)
print(sol.gene_type)
print(sol.gene_space)

pop = sol.population(5)
pprint(pop)
