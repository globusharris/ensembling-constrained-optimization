import cvxpy as cp
import numpy as np
from sklearn.covariance import empirical_covariance


def variance_constrained_allocation(predictions, d, tau, covariance, variance_limit):
    """
    Given a prediction for a single context, outputs the optimal solution to the
    long-only variance-constrained optimization.

    predictions: d-dimensional array of predictions from a single context
    d: dimension of labels
    tau: number of decimal places of the discretization
    covariance: covariance matrix of the d outcomes
    variance_limit: scalar variance limit
    """
    x = cp.Variable(d)
    objective = cp.Maximize(x @ predictions)

    # TODO see if can discretize the solution via constraints in optimization
    #discretization_set = np.array([0, 1, 2])
    #discretization_set = set([i / (10**tau) for i in range(10**tau+1)])
    #discretization_constraint = cp.constraints.finite_set.FiniteSet(x, discretization_set)
    #constraints = [discretization_constraint, x>=0, x @ covariance @ x.T <= variance_limit]

    constraints = [x<=1, x>=0, x @ covariance @ x.T <= variance_limit]
    constraints.append(sum(x)==1)

    prob = cp.Problem(objective, constraints)

    obj = prob.solve()
    solution = x.value
    true_risk = x.value @ covariance @ x.value.T
    risk_limit = variance_limit

    rounded_solution = np.round(solution, tau) # for now, discretizing the opt solution to tau decimal places

    # debugging
    #print("Optimal objective value", obj)
    #print("Optimal variable", solution)
    #print("true risk: ", true_risk)
    #print("risk limit: ", variance_limit)
    #print("rounded solution: ", rounded_solution)

    return obj, rounded_solution, true_risk, risk_limit


def main():

    ### testing with random inputs
    predictions = [0, 1, 0, 1, 0, 0, 1]
    dim = 7
    tau = 2 # number of decimal places to discretize to
    variance_limit = 1
    labels = np.random.randn(10, 7)
    ###

    covariance = empirical_covariance(labels, assume_centered=False) # gets the empirical variance from labels
    obj, solution, true_risk, risk_limit = variance_constrained_allocation(predictions, dim, tau, covariance, variance_limit)

if __name__ == "__main__":
    main()
