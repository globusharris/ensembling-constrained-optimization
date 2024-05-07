import numpy as np
import cvxpy as cp
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

    def run(preds):
        return -1

class Simplex(Policy):

    def __init__(self, dim, model):
        Policy.__init__(self, dim, model)
        self.name = "simplex"
        self.coordinate_values = [1] 
        self.n_vals = len(self.coordinate_values)
    
    def run(self, xs):
        preds = self.model(xs)
        allocation = np.zeros((len(preds), self.dim))
        max_coord = np.argmax(preds, axis=1)
        allocation[np.arange(len(preds)), max_coord] = np.ones(len(preds))
        return allocation

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

class VarianceConstrained(Policy):

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
        it summing to 1 across all coordinates and, if w is the policy vector, wCw^T <= alpha, where
        C is the (empirical) covariance matrix of the true ys.
        
        It runs this optimization problem separately for each prediction vector that is input. 
        """

        ### Question: is there a way to run this in parallel for each of the predictions that is better?
        # I feel like there must be. 

        allocation = np.zeros((len(preds), self.dim))
        for i in range(len(preds)):
            x = cp.Variable(self.dim)
            objective = cp.Maximize(x @ preds[i])
            # for some reason, the last constraint isn't actually constraining things to be less than 1? 
            constraints = [x<=1, x>=0, x @ self.covariance @ x.T <= self.var_limit, x @ np.ones(self.dim) == 1]
            
            prob = cp.Problem(objective, constraints)
            obj = prob.solve()  
            solution = x.value
            n_decimals = int(abs(np.log10(self.gran)))
            truncated_solution = (solution*10**n_decimals//1)/(10**n_decimals)
            normalized_solution = truncated_solution/sum(truncated_solution)
            allocation[i] = normalized_solution
        return allocation

