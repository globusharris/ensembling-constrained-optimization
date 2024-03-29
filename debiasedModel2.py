import numpy as np

class Debiased_model:
    
    def __init__(self, depth, policy, prediction_dim, init_model, train_x):
        self.depth = depth #number of rounds of debiasing 
        self.policy = policy # policy used for optimization, should be a class
        self.prediction_dim = prediction_dim #dimension of predictions
        
        # initializing the model 
        self.init_model = init_model
        self.curr_preds = self.init_model(train_x)
        self.predictions_by_round = [np.copy(self.curr_preds)] 

        # bookkeeping for final predictor
        self.debias_conditions = []
        self.bias_array = [] 
        self.indices_by_round = [] #for debugging

        self.n_conditions = None 
        self.halting_cond = 0

    def predict(self, xs):

        preds = self.init_model(xs)
        
        for t in range(self.depth):        
            coord, val, db_policy = self.debias_conditions[t]
            # get policy induced by current round's predictions 
            curr_policy = db_policy.run_given_preds(preds)
            # pull out the indices where the policy induces val at the target coordinate 
            indices = np.arange(len(curr_policy))[curr_policy[:,coord] == val]
            bias = self.bias_array[t]
            preds[indices] -= bias 
        
        return preds

    def debias(self, train_y, debiasing_policy):
        
        # set early stopping condition for if last #debiasing-conditions rounds all had 0 bias
        self.n_conditions = len(debiasing_policy.coordinate_values)*debiasing_policy.dim
        
        # run debiasing until debiased or reached max depth
        for t in range(self.depth):
            
            # event to bucket with
            coord = t%debiasing_policy.dim
            val = debiasing_policy.coordinate_values[t%len(debiasing_policy.coordinate_values)]
            self.debias_conditions.append([coord,val,debiasing_policy])
            
            # get policy induced by current round's predictions
            curr_policy = debiasing_policy.run_given_preds(self.curr_preds)
            # pull out the indices where the policy induces val at the target coordinate 
            indices = np.arange(len(curr_policy))[curr_policy[:,coord] == val]
            
            self.indices_by_round.append(indices) #for debugging
            
            if len(indices)!=0:
                # calculate bias on those indices 
                bias = np.mean(self.curr_preds[indices], axis=0) - np.mean(train_y[indices], axis=0)
                # zeroing out floating point issues
                if np.all(np.isclose(bias, np.zeros(len(bias)), atol=1e-8)):
                    bias = np.zeros(len(bias))
                self.bias_array.append(bias)
                self.curr_preds[indices] -= bias 
                # storing predictions over rounds for fun.
                self.predictions_by_round.append(np.copy(self.curr_preds)) #slicing to force new copy that isn't mutable
            else:
                # to keep bookkeeping consistent for halting cond, storing 0 even if bucket empty
                self.bias_array.append(np.zeros(self.prediction_dim))

            if self._halt(t):
                self.depth = t
                break 
 
    def _halt(self, t):
        # halt if no bias found on last set of rounds

        if np.all(self.bias_array[t] == 0.0):
            self.halting_cond += 1
            if self.halting_cond == self.n_conditions:
                print("Halting early at round ", t+1)
                return True
            else: 
                return False
        else:
            self.halting_cond = 0
            return False


