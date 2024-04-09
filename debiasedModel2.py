import numpy as np
from sklearn.metrics import mean_squared_error as mse

class Debiased_model:
    
    def __init__(self, max_depth, prediction_dim, init_model, train_x, train_y):
        self.max_depth = max_depth #number of rounds of debiasing 
        self.prediction_dim = prediction_dim #dimension of predictions
        
        # initializing the model 
        self.init_model = init_model
        self.curr_preds = self.init_model(train_x)
        self.predictions_by_round = [np.copy(self.curr_preds)] 
        self.curr_depth = 0
        self.train_x = train_x

        # bookkeeping for final predictor
        self.debias_conditions = []
        self.bias_array = [] 
        self.indices_by_round = [] #for debugging
        self.mses_by_policy = [mse(train_y, self.curr_preds, multioutput='raw_values')]

        self.n_conditions = None 
        self.halting_cond = 0
        self.maxed_depth = False #flag for if max depth was reached

    def predict(self, xs):

        preds = self.init_model(xs)
        
        for t in range(self.curr_depth):        
            coord, val, db_policy = self.debias_conditions[t]
            # get policy induced by current round's predictions 
            curr_policy = db_policy.run(xs)
            # pull out the indices where the policy induces val at the target coordinate 
            indices = np.arange(len(curr_policy))[curr_policy[:,coord] == val]
            bias = self.bias_array[t]
            preds[indices] -= bias 
        
        return preds

    def debias(self, train_y, debiasing_policy, self_debias=False):
        
        # if not debiasing wrt own model, then can evaluate the debiasing policy once. Otherwise, the policy 
        # changes each round so have to update within the debiasing
        if not self_debias:
            curr_policy = debiasing_policy.run(self.train_x)
        
        # set early stopping condition for if last #debiasing-conditions rounds all had 0 bias
        self.n_conditions = len(debiasing_policy.coordinate_values)*debiasing_policy.dim
        
        # run debiasing until debiased or reached max depth
        t = self.curr_depth
        i=0 # indexing for iterating through coordinates and values
        while t <= self.max_depth:
            # event to bucket with
            n_by_coord = len(debiasing_policy.coordinate_values)
            coord = (t//n_by_coord) % self.prediction_dim
            val = debiasing_policy.coordinate_values[i%n_by_coord]
            self.debias_conditions.append([coord,val,debiasing_policy])
            
            # get debiasing policy's predictions
            if self_debias:
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

            if self._simple_halt(t):
                # recalculate mse: 
                self.mses_by_policy.append(mse(train_y, self.curr_preds, multioutput='raw_values'))

                ## can get rid of the last n_conditions rounds because bias was 0 for those
                self.curr_depth = t - self.n_conditions + 1
                self.bias_array = self.bias_array[:-self.n_conditions]
                self.predictions_by_round = self.predictions_by_round[:-self.n_conditions]
                self.indices_by_round = self.indices_by_round[:-self.n_conditions]
                self.halting_cond = 0 # reseting for next round of debiasing
                break 

            t += 1
            i += 1
    
    def _simple_halt(self, t):
        # halt if no bias found on last set of rounds
        if t == self.max_depth:
                print("Maximal depth reached; halting debiasing .")
                self.maxed_depth = True
                return True 
        
        if np.all(self.bias_array[t] == 0.0):
            self.halting_cond += 1
            if self.halting_cond == self.n_conditions:
                return True
            else: 
                return False
        else:
            self.halting_cond = 0
            return False
    
    def iterative_debias(self, train_y, policies, own_policy, depth, tolerance):
        ### given a list of policies, get to point where model is self-consistent and debiased wrt all of them.
        i = 0
        n_policies = len(policies) + 1 #number of things we're debiasing wrt, including self-consistency check
        self_consistency = False
        for t in range(depth):

            if self.maxed_depth:
                break

            if self_consistency:
                # run self-consistency check
                own_policy.model = self
                self.debias(train_y, own_policy, self_debias=True)
                self_consistency=False
                
                if self._iterative_halt_condition(n_policies, tolerance):
                    print("Hit tolerance; halting debiasing.")
                    break

            else:
                # debiasing wrt other policies
                self.debias(train_y, policies[i%len(policies)], self_debias=False)
                i += 1
                if i%len(policies)==0:
                    self_consistency=True
            
        return None
    
    def _iterative_halt_condition(self, n_policies,tolerance):
        # check the last k iterations' improvement in mse
        # halt if all of them were smaller than tolerance
        improvement = -1*np.diff(self.mses_by_policy[-(n_policies+1):], axis=0) #+1 bc need to get difference w previous round
        if np.max(improvement) < tolerance:
            return True
        return False


class Ensembled_model:

    def __init__(self, max_depth, prediction_dim, init_model, init_policy, train_x, train_y):
        self.max_depth = max_depth #number of rounds of debiasing 
        self.prediction_dim = prediction_dim #dimension of predictions
        
        # initializing the model 
        self.init_model = init_model
        self.policy = init_policy
        self.curr_preds = self.init_model(train_x)
        self.curr_depth = 0
        self.train_x = train_x

        # bookkeeping for final predictor
        self.predictions_by_round = [np.copy(self.curr_preds)] 
        self.debias_conditions = []  # length t; each index has [coordinate, value, hypotheses, policies], where hypotheses and policies are arrays of length k 
        self.bias_array = []  # length t; at each index is a numpy array of shape k x pred_dim 
        self.indices_by_round = [] #for debugging
        self.mses_by_policy = [mse(train_y, self.curr_preds, multioutput='raw_values')]

        self.n_conditions = None 
        self.halting_cond = 0
        self.maxed_depth = False #flag for if max depth was reached
    
    def predict(self, xs):

        """
        xs: n x prediction_dim array of values to predict on.

        Let k be the number of policies that debiasing was done wrt
        """

        preds = self.init_model(xs)
        
        for t in range(self.curr_depth):        
            # because we need to evaluate which policy has the highest self-evaluated revenue, 
            # need to know both what each policy is *and* what the hypothesis used to induce that
            # policy was, so that we can evaluate this revenue expression

            coord, val, hypotheses, policies = self.debias_conditions[t]

            pred_by_h = [h(xs) for h in hypotheses] # array of length k, where each entry is of shape n x pred_dim
            policies_by_h = [policies[i].run_given_preds(pred_by_h[i]) for i in range(len(policies))] # array of length k, where each entry is of shape n x pred_dim
            self_assessed_revs =  np.array([np.einsum('ij,ij->i', pred_by_h[i], policies_by_h[i]) for i in range(len(policies))]) # array of length k, where each entry is of shape n, and is dot product of pred and policy vector
            maximal_policy = np.argmax(self_assessed_revs, axis=0) # length n; returns index of the maximal policy
            
            # get policy induced by current round's predictions 
            curr_policy = self.policy.run_given_preds(preds)
            # pull out the indices where the policy induces val at the target coordinate 
            indices = np.arange(len(curr_policy))[curr_policy[:,coord] == val]
            
            # if coord, val matches, subtract off whichever bias term 
            # make sure bias is np array or this doesn't work
            preds[indices] -= self.bias_array[t][maximal_policy[indices]] 
        
        return preds

    def debias_wrt_max(self, train_y, all_policies):
        
        t = self.curr_depth
        i=0 # indexing for iterating through coordinates and values
        while t <= self.max_depth:
            coord = i%self.prediction_dim
            val = self.policy.coordinate_values[i%len(self.policy.coordinate_values)]
            print(coord, val)


