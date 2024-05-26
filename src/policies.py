import numpy as np
import gurobipy as gp
from sklearn.covariance import empirical_covariance

class Policy:

    def __init__(self, dim, model):
        """
        name: string value describing what type of allocation policy
        gran: granularity used in generating the policy (e.g. if 0.1, can
                allocate either 0, 0.1, 0.2,...,1 to any coordinate)
        coordinate_values: bookkeeping for conditioning events; array of possible
                values each coordinate may take. E.g. [1] in case of simplex. 
                * technically [0,1] in case of simplex but we just consider [1] since
                we only really care about bias in the case where an allocation actually happens.
        n_vals: length of coordinate values array
        dim: dimension of policy decisions. Should be same as the prediction dimension. 
        """
        self.name = None
        self.gran = None
        self.coordinate_values = None
        self.n_vals = None
        self.dim = dim
        self.model = model

class Simplex(Policy):
    """
    Simple policy which always picks the largest coordinate of the prediction vector and puts weight 1 on that
    and 0 elsewhere, which is the solution to the unconstrained optimization of max policy \dot predictions on the
    probability simplex. It runs quickly and without any optimization library, so useful for troubleshooting. 
    """

    def __init__(self, dim, model):
        Policy.__init__(self, dim, model)
        self.name = "simplex"
        self.coordinate_values = [1] 
        self.gran = 0.1 # this is meaningless for this policy
        self.n_vals = len(self.coordinate_values)

    def run_given_preds(self, preds):
        # expects numpy matrix of predictions, where 1 row corresponds to a single vector of predictions
        """
        preds: array of predictions, where one row corresponds to a single vector of predictions.
        return: allocation vector for each prediction. For optimization over simplex with no additional constraints,
        this corresponds to putting all of the allocation's weight on the maximal coordinate of each prediction.
        """
        allocation = np.zeros((len(preds), self.dim))
        max_coord = np.argmax(preds, axis=1)
        allocation[np.arange(len(preds)), max_coord] = np.ones(len(preds))
        return allocation

class Linear(Policy):
    """
    Extremely simple linear optimization. 
    Input linear constraints are a m x pred_dim matrix of constraint conditions and max_val
    is an m-dimensional vector. So e.g. if 
        linear_constraint = [[0,1,1], [1,1,0]]
    and 
        max_val = [0.1,0.2]
    Then this corresponds the following optimization problem:
    max w \dot v 
    st
        w_2 + w_3 < 0.1,
    and
        w_1 + w_2 < 0.2,
    and
        w_i \in [0,1].
    """
    def __init__(self, dim, model, gran, linear_constraint, max_val):
        Policy.__init__(self, dim, model)
        self.name = "linear-min"
        self.gran = gran
        self.coordinate_values = np.arange(0,1,gran)
        self.n_vals = len(self.coordinate_values)
        self.linear_constraint = linear_constraint
        self.max_val = max_val
    
    def run_given_preds(self, preds):
        """
        preds: array of predictions, where one row corresponds to a single vector of predictions.
        return: allocation vector for each prediction.
        """
        allocation = np.zeros((len(preds), self.dim))
        # the next three lines suppress gurobi's printout of results.
        # (because this is running in a loop and will be called externally many times)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        # Running the allocation for each row of predictions
        for i in range(len(preds)):
            m = gp.Model(env=env)
            x = m.addMVar(self.dim, lb=0.0, ub=1.0)
            m.setObjective(x @ preds[i], gp.GRB.MAXIMIZE)
            m.addMConstr(self.linear_constraint, x, '<', self.max_val)
            m.optimize()
            allocation[i]=x.X
            # next line frees all resources associated w model object. Gurobi has some issues w memory leaks,
            # which was leading to very high resource use when optimization was run repeatedly. 
            # originally implemented w cvxpy, which had this issue to a much greater extent. 
            m.dispose() 
        return allocation

class VarianceConstrained(Policy):

    """
    Outputs policies that are constrained such that the weight vector sums to 1 and is bounded 
    by the empirical covariance matrix of the labels. 
    """
    def __init__(self, dim, model, gran, var_limit, ys):
        Policy.__init__(self, dim, model)
        self.name = "minimize-variance"
        self.gran = gran
        self.var_limit = var_limit
        self.ys = ys
        self.covariance = empirical_covariance(ys, assume_centered=False)
        self.coordinate_values = np.arange(0,1,gran)
        self.n_vals = len(self.coordinate_values)
 
    
    def run_given_preds(self, preds):
        """
        preds: array of predictions, where one row corresponds to a single vector of predictions.
        return: allocation vector for each prediction.
        
        The allocation vector is computed by constraining the problem to maximize the policy subject to
        it summing to 1 across all coordinates, each allocation term being bounded by 0 and 1, 
        and, if w is the policy vector, wCw^T <= alpha, where C is the (empirical) covariance matrix of the labels.
        
        It runs this optimization problem separately for each prediction vector that is input. 
        """
        allocation = np.zeros((len(preds), self.dim))
        # the next three lines suppress gurobi's printout of results.
        # (because this is running in a loop and will be called externally many times)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        for i in range(len(preds)):
            m = gp.Model(env=env)
            x = m.addMVar(self.dim, lb=0.0, ub=1.0)
            m.setObjective(x @ preds[i], gp.GRB.MAXIMIZE)
            m.addConstr(x.sum() == 1)  # allocation forms a distribution which sums to 1
            m.addConstr(x @ self.covariance @ x <= self.var_limit)
            m.optimize()
            allocation[i]=x.X
            # next line frees all resources associated w model object. Gurobi has some issues w memory leaks,
            # which was leading to very high resource use when optimization was run repeatedly. 
            # originally implemented w cvxpy, which had this issue to a much greater extent. 
            m.dispose()
        return allocation

