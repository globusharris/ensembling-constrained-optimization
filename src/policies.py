import numpy as np
import cvxpy as cp
from sklearn.covariance import empirical_covariance
import warnings
from multiprocessing import Pool

# supress future warnings from cvxpy
warnings.simplefilter(action='ignore', category=FutureWarning)

# Below code for the multithreaded version of the code, which doesn't seem to like functions being inside of objects. 
# def covariance_constrained_optimization_problem(args):
#     pred, covariance, var_limit = args
#     dim = len(pred)
#     x = cp.Variable(dim)
#     objective = cp.Maximize(x @ pred)
#     constraints = [x<=1, # have to allocate between 0 and 1
#                     x>=0, 
#                     x @ np.ones(dim) == 1, # allocation forms a distribution which sums to 1
#                     cp.quad_form(x, covariance) <= var_limit  # allocation bounded by covariance matrix of ys
#                     ]
#     prob = cp.Problem(objective, constraints)
#     prob.solve(solver=cp.GUROBI, verbose=False) 
#     return x.value

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
        self.gran = 0.1 # this is meaningless for this policy since don't need to worry about binning here, but need in order to match types of other policies
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

class LinearMin(Policy):

    def __init__(self, dim, model, gran, ys):
        Policy.__init__(self, dim, model)
        self.name = "linear-min"
        self.gran = gran
        self.ys = ys
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

        Pooling causing problems w suppressing stdout, and also for some reason slower than alternative, so tossing for now. 
        """
        # with Pool() as pool:
        #     results = pool.map(covariance_constrained_optimization_problem, [(pred, self.covariance, self.var_limit) for pred in preds])
        # return np.array(results)
        allocation = np.zeros((len(preds), self.dim))

        for i in range(len(preds)):
            x = cp.Variable(self.dim)
            objective = cp.Minimize(x @ preds[i])
            constraints = [x<=1, # have to allocate between 0 and 1
                            x>=0, 
                            x @ np.ones(self.dim) == 1 # allocation forms a distribution which sums to 1
                            # cp.quad_form(x,self.covariance) <= self.var_limit  # allocation bounded by covariance matrix of ys
                            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.GUROBI, verbose=False) 
            allocation[i] = x.value
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
        it summing to 1 across all coordinates, each allocation term being bounded by 0 and 1, 
        and, if w is the policy vector, wCw^T <= alpha, where C is the (empirical) covariance matrix of the labels.
        
        It runs this optimization problem separately for each prediction vector that is input. 

        Pooling causing problems w suppressing stdout, and also for some reason slower than alternative, so tossing for now. 
        """
        # with Pool() as pool:
        #     results = pool.map(covariance_constrained_optimization_problem, [(pred, self.covariance, self.var_limit) for pred in preds])
        # return np.array(results)
        allocation = np.zeros((len(preds), self.dim))

        for i in range(len(preds)):
            x = cp.Variable(self.dim)
            objective = cp.Maximize(x @ preds[i])
            constraints = [x<=1, # have to allocate between 0 and 1
                            x>=0, 
                            x @ np.ones(self.dim) == 1, # allocation forms a distribution which sums to 1
                            cp.quad_form(x,self.covariance) <= self.var_limit  # allocation bounded by covariance matrix of ys
                            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.GUROBI, verbose=False) 
            allocation[i] = x.value
        return allocation

class LinearConstrained(Policy):
    pass

class Bipartite(Policy):
    pass
   
# class ElectricTransformer(Policy):
    
#     def __init__(self, dim, model, gran, alloc_limit):
#         Policy.__init__(self, dim, model)
#         self.name = "electric-transformer"
#         self.gran = gran
#         self.alloc_limit = alloc_limit
#         self.coordinate_values = np.arange(0,1,gran)
#         self.n_vals = len(self.coordinate_values)
    
#     def run_given_preds(self, preds):
#         allocation = np.zeros((len(preds), self.dim))
#         for i in range(len(preds)):
#             if i%1000==0:
#                 print('i', i)
#             x = cp.Variable(self.dim)
#             objective = cp.Maximize(x @ preds[i])

#             constraints = [
#                 x<=1, # decision variables bounded between 0 and 1
#                 x>=0, 
#                 cp.sum(x) == 1,
#                 cp.multiply(x, preds[i]) <= self.alloc_limit
#                 ]
#                 # want a constraint that disincentivizes allocations for too big of values...
#                 # the above doesn't really do this..
#             prob = cp.Problem(objective, constraints)
#             prob.solve()
#             sol = x.value
#             allocation[i] = sol
#         return allocation
            


