from ortools.linear_solver import pywraplp

ORTOOLS_SOLVERS = [
    "CLP_LINEAR_PROGRAMMING", "CLP",
    "CBC_MIXED_INTEGER_PROGRAMMING", "CBC",
    "GLOP_LINEAR_PROGRAMMING", "GLOP",
    "BOP_INTEGER_PROGRAMMING", "BOP",
    "SAT_INTEGER_PROGRAMMING", "SAT", "CP_SAT",
    "SCIP_MIXED_INTEGER_PROGRAMMING", "SCIP",
    "GUROBI_LINEAR_PROGRAMMING", "GUROBI_LP",
    "GUROBI_MIXED_INTEGER_PROGRAMMING", "GUROBI", "GUROBI_MIP",
    "CPLEX_LINEAR_PROGRAMMING", "CPLEX_LP",
    "CPLEX_MIXED_INTEGER_PROGRAMMING", "CPLEX", "CPLEX_MIP",
    "XPRESS_LINEAR_PROGRAMMING", "XPRESS_LP",
    "XPRESS_MIXED_INTEGER_PROGRAMMING", "XPRESS", "XPRESS_MIP",
    "GLPK_LINEAR_PROGRAMMING", "GLPK_LP",
    "GLPK_MIXED_INTEGER_PROGRAMMING", "GLPK", "GLPK_MIP",
]

def check_solvers():
    for sname in ORTOOLS_SOLVERS:
        solver = pywraplp.Solver.CreateSolver(sname)
        if solver is None:
            print(f"ERROR: {sname}")
        else:
            print(f"_____: {sname}")
    # end
# end


def example():
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return

    # Create the variables x and y.
    x = solver.NumVar(0, 1, "x")
    y = solver.NumVar(0, 2, "y")

    print("Number of variables =", solver.NumVariables())

    # Create a linear constraint, 0 <= x + y <= 2.
    ct = solver.Constraint(0, 2, "ct")
    ct.SetCoefficient(x, 1)
    ct.SetCoefficient(y, 1)

    print("Number of constraints =", solver.NumConstraints())

    # Create the objective function, 3 * x + y.
    objective = solver.Objective()
    objective.SetCoefficient(x, 3)
    objective.SetCoefficient(y, 1)
    objective.SetMaximization()

    print(f"Solving with {solver.SolverVersion()}")
    solver.Solve()

    print("Solution:")
    print("Objective value =", objective.Value())
    print("x =", x.solution_value())
    print("y =", y.solution_value())


def main():
    check_solvers()
    example()
    pass

if __name__ == "__main__":
    main()
