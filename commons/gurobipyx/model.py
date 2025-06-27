import gurobipy as gp



def addArrayVar(model: gp.Model, shape: int|tuple[int, int], lb=0.0, ub=None, obj=0.0, vtype=None, name=""):
    indices = []
    if isinstance(shape, int):
        shape = (shape,)

    assert isinstance(shape, tuple)

    if len(shape) == 1:
        n = shape[0]
        for i in range(n):
            indices.append(i)
    elif len(shape) == 2:
        n, m = shape
        for i in range(n):
            for j in range(m):
                indices.append((i,j))
    else:
        raise ValueError(f"Unsupported shape {shape}")

    return model.addVars(indices, lb=lb, ub=ub, vtype=vtype, name=name)
# end


def addConstrs(model: gp.Model, constrs: list, name=None):
    if not isinstance(constrs, (list, tuple)):
        c = constrs
        cid = model.addConstr(c, name=("" if name is None else name))
        cids = [cid]
    else:
        cids = []
        index = 0
        for c in constrs:
            cid = model.addConstr(c, name="" if name is None else f"{name}[{index}]")
            index += 1
            cids.append(cid)
    return cids
# end
