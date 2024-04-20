import numpy as np
import copy
from sklearn.metrics import mean_squared_error as mse

class bbDebias:
    """
    "Bias Bounty" style method of iteratively debiasing a predictor with respect to a series of policies
    """
    
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
        self.indices_by_round = [] #for debugging/tracking size of groups, etc
        self.mses_by_policy = [mse(train_y, self.curr_preds, multioutput='raw_values')]

        self.n_conditions = None 
        self.halting_cond = 0
        self.maxed_depth = False #flag for if max depth was reached

    def predict(self, xs):

        """
        Use debiased predictor to get predictions on xs.
        """

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
    
    def debias(self, train_y, policies, own_policy, depth, tolerance):
        """
        Given an input list of policies, debias initial policy until either maximal depth is reached or model is self-consistent
        and unbiased with respect to all of the policies.
        """
        i = 0
        n_policies = len(policies) + 1 #number of things we're debiasing wrt, including self-consistency check
        self_consistency = False
        for t in range(depth):

            if self.maxed_depth:
                break

            if self_consistency:
                # run self-consistency check
                own_policy.model = self
                self.debias_helper(train_y, own_policy, self_debias=True)
                self_consistency=False
                
                if self._halt(n_policies, tolerance):
                    print("Hit tolerance; halting debiasing.")
                    break

            else:
                # debiasing wrt other policies
                self.debias_helper(train_y, policies[i%len(policies)], self_debias=False)
                i += 1
                if i%len(policies)==0:
                    self_consistency=True
            
        return None
    
    def _halt(self, n_policies,tolerance):
        """
        Halting condition for the debiasing process. It checks the last k iterations' improvement in MSE, and
        halts if all of them were smaller than the tolerance. 

        Note: k = number of policies which you are debiasing with respect to. If you only checked the previous round, rather than
        the last k, you'd only know if the last policy you debiased with respect to led to improvement in squared error. It could
        be that you can make no more improvement with respect to that policy, but that there is some other policy which does
        lead to improvement, hence the condition checking all of them.
        """
        improvement = -1*np.diff(self.mses_by_policy[-(n_policies+1):], axis=0) #+1 bc need to get difference w previous round
        if np.max(improvement) < tolerance:
            return True
        return False

    def debias_helper(self, train_y, debiasing_policy, self_debias=False):
        """
        Debiasing on a single policy. This could either be own policy, in which case you have to enforce self-consistency,
        or it could be anothers'. 

        ToDo? Move self-consistency check elsewhere? This code is quite messy. 
        """
        
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
            
            if self_debias:
                # if doing self-debiasing, create a new version of the debiasing policy that actually uses current predictions
                debiasing_policy.model = self.predict

            self.debias_conditions.append([coord,val,copy.deepcopy(debiasing_policy)])
            
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
        """
        Halting condition for debiasing a single policy. If, looking at all coord x val pairs, no bias was found, then halts.
        Also halts if maximal depth has been reached. 
        To Do: should add tolerance condition to this. 
        """
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
    
    





            
            


