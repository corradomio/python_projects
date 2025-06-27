import numpy as np
import gurobipy as gp
import gurobipyx as gpx
import gurobipyx.lin
import gurobipyx.constr
from gurobipy import GRB
from stdlib.jsonx import load


def my_problem():
    model = gp.Model("spare_distribution")

    # https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/examples.html
    # time limit
    # model.setParam('TimeLimit', 60)
    model.Params.TimeLimit = 120
    # model.Params.MIPGap = 0.070

    jdata = load(r"D:\Projects.github\python_projects\check_bt_spare_distribution\data_synth\optim\optim_problem_100.json")
    A = jdata["A"]
    R = jdata["R"]
    D = np.array(jdata["D"])

    n = len(A)
    m = len(R)

    # indices = []
    # for i in range(n):
    #     for j in range(m):
    #         indices.append((i,j))

    # tvars = model.addVars(indices, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="t" )
    # xvars = model.addVars(indices, lb=0, ub=1, vtype=GRB.BINARY, name="x" )

    tvars = gpx.addArrayVar(model, (n,m), lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="t")
    xvars = gpx.addArrayVar(model, (n,m), lb=0, ub=1, vtype=GRB.BINARY, name="x")

    #
    # fitness function
    #
    # fun = xvars[(0,0)]*D[0,0]
    # for i in range(n):
    #     for j in range(m):
    #         if i==0 and j == 0: continue
    #         fun = fun + xvars[i,j]*D[i,j]

    fun = gpx.lin.sum_hadamard(xvars, D)
    # fun = gpx.lin.sum_hadamard(D, xvars)
    # fun = gpx.lin.sum_hadamard(xvars, tvars)

    #
    # constraints
    #

    # # constrains t/A
    # for i in range(n):
    #     ct = tvars[(i,0)]
    #     for j in range(1,m):
    #         ct = ct + tvars[i,j]
    #
    #     model.addConstr(ct <= A[i], f"cA_{i}")


    # cA = gpx.constr.sum_row_leq(tvars, A)
    # --
    trows = gpx.lin.sum_rows(tvars)
    cA = gpx.constr.leq(trows, A)

    gpx.addConstrs(model,cA, name="cA")

    # constrains t/R
    # for j in range(m):
    #     ct = tvars[(0,j)]
    #     for i in range(1, n):
    #         ct = ct + tvars[i,j]
    #
    #     model.addConstr(ct == R[j], f"cR_{j}")

    # cR = gpx.constr.sum_col_eq(tvars, R)
    # --
    tcols = gpx.lin.sum_cols(tvars)
    cR = gpx.constr.eq(tcols, R)

    gpx.addConstrs(model, cR, name="cR")

    # for i in range(n):
    #     for j in range(m):
    #         model.addConstr(tvars[i,j] <= xvars[i,j]*A[i], f"cA_{i}_{j}")
    #         model.addConstr(tvars[i,j] <= xvars[i,j]*R[j], f"cR_{i}_{j}")

    xA = gpx.lin.broadcast_col(xvars, A)
    bR = gpx.constr.leq(tvars, xA)
    gpx.addConstrs(model, bR, name="bR")

    xR = gpx.lin.broadcast_row(xvars, R)
    bA = gpx.constr.leq(tvars, xR)
    gpx.addConstrs(model, bA, name="bA")

    # Set objective
    model.setObjective(fun, GRB.MINIMIZE)

    model.write("model.lp")

    # Optimize model
    model.optimize()

    print(f"Obj: {model.ObjVal:g}")

    pass



def example():

    try:
        # Create a new model
        model = gp.Model("mip1")

        # Create variables

        x = model.addVar(vtype=GRB.BINARY, name="x")
        y = model.addVar(vtype=GRB.BINARY, name="y")
        z = model.addVar(vtype=GRB.BINARY, name="z")

        # Set objective
        model.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

        # Add constraint: x + 2 y + 3 z <= 4
        model.addConstr(x + 2 * y + 3 * z <= 4, "c0")

        # Add constraint: x + y >= 1
        model.addConstr(x + y >= 1, "c1")

        # Optimize model
        model.optimize()

        for v in model.getVars():
            print(f"{v.VarName} {v.X:g}")

        print(f"Obj: {model.ObjVal:g}")

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

    except AttributeError:
        print("Encountered an attribute error")


def main():
    # example()
    my_problem()
    pass

# end


if __name__ == "__main__":
    main()
