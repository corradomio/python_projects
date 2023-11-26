# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random non-trivial quadratic program.
m = 15
n = 10
p = 5
np.random.seed(1)
P = np.random.randn(n, n)
P = P.T@P
q = np.random.randn(n)
G = np.random.randn(m, n)
h = G@np.random.randn(n)
A = np.random.randn(p, n)
b = np.random.randn(p)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T@x),
                 [G@x <= h,
                  A@x == b])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)

# The optimal value is 86.89141585569918
# A solution x is
# [-1.68244521  0.29769913 -2.38772183 -2.79986015  1.18270433 -0.20911897
#  -4.50993526  3.76683701 -0.45770675 -3.78589638]
# A dual solution corresponding to the inequality constraints is
# [ 0.          0.          0.          0.          0.         10.45538054
#   0.          0.          0.         39.67365045  0.          0.
#   0.         20.79927156  6.54115873]
