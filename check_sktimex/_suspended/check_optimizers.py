
import numpy as np
from hyperactive.opt.gfo import HillClimbing

# Define objective function (maximize)
def objective(params):
    x, y = params["x"], params["y"]
    return -(x**2 + y**2)  # Negative paraboloid, optimum at (0, 0)

# Define search space
search_space = {
    "x": np.arange(-5, 5, 0.1).tolist(),
    "y": np.arange(-5, 5, 0.1).tolist(),
}

# Run optimization
optimizer = HillClimbing(
    search_space=search_space,
    n_iter=100,
    experiment=objective,
)
best_params = optimizer.solve()

print(f"Best params: {best_params}")
