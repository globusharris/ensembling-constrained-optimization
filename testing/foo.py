import cvxpy as cp

def foofoo():
    print('wee')
    return 5

def optimization(y):
    x = cp.Variable(3)
    objective = cp.Maximize(x @ y)
    constraints = [x<=1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)
