import numpy as np

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

