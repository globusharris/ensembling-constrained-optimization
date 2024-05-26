import numpy as np


"""
A bunch of initial versions of things, which were implemented stupidly. Keeping around just in case.
Didn't track all models' predictions at once, which made too slow.
"""

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
        self.train_y = train_y

        # bookkeeping for final predictor
        self.predictions_by_round = [np.copy(self.curr_preds)] 
        self.debias_conditions = []  # length t; each index has [coordinate, value, hypotheses, policies], where hypotheses and policies are arrays of length k 
        self.bias_array = []  # length t; at each index is a numpy array of shape k x pred_dim 
        self.indices_by_round = [] #for debugging
        self.mses_by_policy = [mse(train_y, self.curr_preds, multioutput='raw_values')]
        self.maximal_policy_by_round = []

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
            # print(indices)
            # print(maximal_policy)
            # print(maximal_policy[indices])
            # print(preds[indices])
            # print(self.bias_array[t][maximal_policy[indices]] )

            # if coord, val matches, subtract off whichever bias term 
            # make sure bias is np array or this doesn't work
            preds[indices] -= self.bias_array[t][maximal_policy[indices]] 
        
        return preds

    def debias_wrt_max(self, hypotheses, policies):

        """
        This is currently running way too slowly if you try to chain policies. Not sure what's going on
        but think has to do w predict function
        ToDo:

        - Add in stopping condition
        - For improved efficiency when iteratively training, write version that can take in current
          hypotheses' predictions instead of rerunning the predict function every time. 
        """

        # For self-consistency, add own model and its induced policy to things to debias wrt
        hypotheses.append(0)
        policies.append(0)
        
        t = self.curr_depth
        i=0 # indexing for iterating through coordinates and values
        while t < self.max_depth:

            # For self-consistency, update current hypothesis and policy
            hypotheses[-1] = self.predict
            self.policy.model = self.predict
            policies[-1] = self.policy

            # Get the conditions to debias with respect to in this round, and store.
            n_by_coord = len(self.policy.coordinate_values)
            coord = (self.curr_depth//n_by_coord) % self.prediction_dim
            val = self.policy.coordinate_values[i%n_by_coord]
            self.debias_conditions.append([coord, val, copy.deepcopy(hypotheses), copy.deepcopy(policies)])

            # calculate bias array conditioned on coord x val and which policy is maximal
            # get predictions for all hypotheses other than own
            pred_by_h = [h(self.train_x) for h in hypotheses[:-1]] # array of length k, where each entry is of shape n x pred_dim
            # add on own hypothesis' most up-to-date predictions for self-consistency 
            pred_by_h.append(self.curr_preds)
            policies_by_h = [policies[i].run_given_preds(pred_by_h[i]) for i in range(len(policies))] # array of length k, where each entry is of shape n x pred_dim
            self_assessed_revs =  np.array([np.einsum('ij,ij->i', pred_by_h[i], policies_by_h[i]) for i in range(len(policies))]) # array of length k, where each entry is of shape n, and is dot product of pred and policy vector
            maximal_policy = np.argmax(self_assessed_revs, axis=0) # length n; returns index of the maximal policy
            self.maximal_policy_by_round.append(maximal_policy)
            # get policy induced by current round's predictions 
            curr_policy = self.policy.run_given_preds(self.curr_preds)
            # calculate the bias over regions where each of the k policies is maximal
            bias_by_policy = []
            for i in range(len(policies)):
                flag = (curr_policy[:,coord] == val) & (maximal_policy == i)
                if sum(flag)!=0:
                    bias = np.mean(self.curr_preds[flag] - self.train_y[flag], axis=0)
                    bias_by_policy.append(bias)
                    self.curr_preds[flag] -= bias
                else:
                    # just store 0 if bucket is empty
                    bias_by_policy.append(np.zeros(self.prediction_dim))
            self.bias_array.append(np.array(bias_by_policy))
            self.predictions_by_round.append(np.copy(self.curr_preds))

            self.curr_depth += 1
            i+=1

    def debias_given_preds(self, predictions_by_h, policies, per_round_max_depth):

        """
        This won't work if you have to look at results on holdout data, but is a workaround for now to deal w fact that 
        other version is taking too long to run.  
        """
        predictions_by_h.append([]) # adding an element for self-consistency checks
        policies.append(self.policy)

        for i in range(per_round_max_depth):

            # Get the conditions to debias with respect to in this round, and store.
            n_by_coord = len(self.policy.coordinate_values)
            coord = (self.curr_depth//n_by_coord) % self.prediction_dim
            val = self.policy.coordinate_values[i%n_by_coord]
            self.debias_conditions.append([coord, val]) # can't track hypotheses bc don't have 'em

            # calculate bias array conditioned on coord x val and which policy is maximal
            # get predictions for all hypotheses other than own
            
            # add on own hypothesis' most up-to-date predictions for self-consistency 
            predictions_by_h[-1] = (self.curr_preds)
            policies_by_h = [policies[i].run_given_preds(predictions_by_h[i]) for i in range(len(policies))] # array of length k, where each entry is of shape n x pred_dim
            self_assessed_revs =  np.array([np.einsum('ij,ij->i', predictions_by_h[i], policies_by_h[i]) for i in range(len(policies))]) # array of length k, where each entry is of shape n, and is dot product of pred and policy vector
            maximal_policy = np.argmax(self_assessed_revs, axis=0) # length n; returns index of the maximal policy
            self.maximal_policy_by_round.append(maximal_policy)
            # get policy induced by current round's predictions 
            curr_policy = self.policy.run_given_preds(self.curr_preds)
            # calculate the bias over regions where each of the k policies is maximal
            bias_by_policy = []
            for i in range(len(policies)):
                flag = (curr_policy[:,coord] == val) & (maximal_policy == i)
                if sum(flag)!=0:
                    bias = np.mean(self.curr_preds[flag] - self.train_y[flag], axis=0)
                    bias_by_policy.append(bias)
                    self.curr_preds[flag] -= bias
                else:
                    # just store 0 if bucket is empty
                    bias_by_policy.append(np.zeros(self.prediction_dim))
            self.bias_array.append(np.array(bias_by_policy))
            self.predictions_by_round.append(np.copy(self.curr_preds))

            self.curr_depth += 1
    
    """
    Original version, iteratively re evaluated things so was too slow. 
    """

    class Debiased_model:

        def __init__(self):
            """
            Not sure if depth should refer to the total number of debiasing rounds or if should 
            have it be a sort of "global" t, where total depth is actually t*gran*pred_dim
            (where you basically do gran*pred_dim different updates at every round)
            """
            self.depth = None #number of rounds of debiasing 
            self.gran = None # granularity of level sets for debiasing (irrelevant for simplex policy)
            self.policy = None # policy used for optimization, should be a class
            self.prediction_dim = None #dimension of predictions
            self.bias_array = None #bookkeeping for bias terms 
            self.training_buckets_indices = None # bookkeeping for bucketing the training data per round of debiasing. stores indices of buckets.
            self.training_buckets_preds = None # bookkeeping for bucketing; stores predictions on training data so don't have to recompute for bias calc
            self.debiasing_cond = None # bookkeeping for bucketing; stores the coordinate that debiasing was done on by round
            self.training_preds = None
            self.n = None #number of training datapoints
            self.model = None #(debiased) model generated

        def debias(self, X_train, Y_train, init_model, max_depth, model_policy, debias_policy, gran=None):
            self.prediction_dim = Y_train.shape[1]
            self.depth = max_depth
            self.policy = model_policy
            self.gran = gran
            self.n = len(X_train)
            self.model = init_model

            # bookkeeping
            n_buckets = self.policy.n_vals  #number of conditioning events that we'll bucket the data into 
            self.training_buckets_indices = [[[] for j in range(n_buckets)] for i in range(self.depth)] # store buckets by index of training data for each round
            self.training_buckets_preds = [[[] for j in range(n_buckets)] for i in range(self.depth)] # store buckets by value of predictions in each round (to avoid recomputing)
            
            self.bias_array = np.zeros((self.depth, self.policy.n_vals, self.policy.dim))
            
            self.training_preds = [[] for i in range(self.depth + 1)] # track all predictions (including initial and final models)
            self.debiasing_cond = [{"coord": None} for i in range(self.depth)] # store the coordinate 

            # Error check: model needs to be cast to floats or else debiasing behaves unexpectedly
            if init_model(X_train[0]).dtype != 'float64':
                raise TypeError("Initial model must cast predictions to floats.")

            # Begin debiasing process:
            t = 0
            early_stop = False
            print("Round of debiasing: ")
            while t < self.depth:
                # need to change halting condition
                if t != 0 and self._halt(t):
                    print()
                    print(f"Model debiasing complete, halting early after round {t}/{max_depth}.")
                    self.training_preds[t] = self.model(X_train)
                    self._truncate_bookkeeping(t)
                    early_stop = True
                    break
                else:   
                # need to loop through the coordinates for debiasing. 
                    for coord in range(self.prediction_dim):
                        if t >= self.depth:
                            break
                        print(t, end="")
                        if self.policy == debias_policy: #if debiasing against self, the model you're debiasing wrt changes every round
                            self._bucket_xs(self.model, debias_policy, X_train, coord, t=t, save=True, self_debias = True)
                        else:
                            self._bucket_xs(self.model, debias_policy, X_train, coord, t=t, save=True)
                        self._update_bias_terms(t, Y_train)
                        
                        if self.policy == debias_policy:
                            self._update_model(self.model, t, debias_policy, self_debias=True)
                        else:
                            self._update_model(self.model, t, debias_policy)
                        t += 1
            if not early_stop:
                self.training_preds[t] = self.model(X_train)
            return self.model

        def _bucket_xs(self, curr_model, debias_policy, xs, coord, t = None, save=False, self_debias = False):
            """
            Helper function which takes the current model and training data, runs the current model
            on this data and then buckets the training data according to the policy's decisions on
            the current model. 

            curr_model: current predictive model
            xs: feature data that is being bucketed
            coord: coordinate to condition on
            val: value to condition on at the specified coordinate
            t: current round of debiasing (for bookkeeping, only needed if using in debiasing process)
            save: Boolean flag. If true, buckets will be stored globally for the training process. Otherwise, 
            bucketed indices are returned but not stored by the model. 

            Note: buckets are fully disjoint since only looking at a single coordinate at a time. 
            """

            # get predictions of the model that will debias wrt and of current model

            if self_debias:
                curr_model_preds = curr_model(xs)
                policy_model_preds = curr_model_preds
            else:
                curr_model_preds = curr_model(xs)
                policy_model_preds = debias_policy.model(xs)
            
            if save:
                # store the predictions and the coordinate the debiasing is run on
                self.training_preds[t] = curr_model_preds
                self.debiasing_cond[t] = coord
            
            # apply policy to each of the predictions 
            policies = self.policy.run_given_preds(policy_model_preds)  
            
            # note that for a single coordinate, the possible values for the policy split 
            # the xs into a disjoint set of buckets/level sets, and hence debiasing can be
            # done in parallel for each of the different values.
            
            if not save:
                n_buckets = self.policy.n_vals
                local_bucket_indices = [[] for i in range(n_buckets)]

            i=0                     
            while i < self.policy.n_vals: # i iterates through possible policy values 

                val = self.policy.coordinate_values[i]
                
                # pull out the indices where the policy induces val at the target coordinate 
                indices = np.arange(len(policies))[policies[:,coord] == val]
                
                if save == True:
                    #record the indices and the current model's predictions at those points
                    self.training_buckets_indices[t][i] = indices
                    self.training_buckets_preds[t][i] = curr_model_preds[indices]
                else:
                    local_bucket_indices[i] = indices
                
                i += 1

            if not save:
                return local_bucket_indices
            else:
                return self.training_buckets_indices  
        
        def _update_bias_terms(self, t, Y_train):
            # update the bias array given current round's predictions

            # calculate the bias terms for this model
            for i in range(len(self.training_buckets_indices[t])):
                bucket = self.training_buckets_indices[t][i]
                if len(bucket)!=0:
                    preds = self.training_buckets_preds[t][i]
                    ys = Y_train[self.training_buckets_indices[t][i]]
                    self.bias_array[t][i] = np.mean(preds, axis=0) - np.mean(ys, axis=0)
                    # Once fully debiased, start getting floating point issues due to the mean calculations.
                    # So have check here for if close to zero, so that halting condition can be raised.
                    if np.all(np.isclose(self.bias_array[t][i], np.zeros(len(self.bias_array[t][i])), atol=1e-8)):
                        self.bias_array[t][i] = np.zeros(len(self.bias_array[t][i]))
                else:
                    # otherwise, never use this term, so not important, but for now filling w zeros
                    # maybe shouldl use nans instead idk.
                    self.bias_array[t][i] = np.zeros(self.prediction_dim)
            
            return self.bias_array
        
        def _update_model(self, curr_model, t, debias_policy, self_debias=False):
            """
            """
            def new_model(xs):
                old_preds = curr_model(xs)

                curr_coord = self.debiasing_cond[t]
                if self_debias:
                    bucket_indices = self._bucket_xs(curr_model, debias_policy, xs, curr_coord, self_debias=True)
                else:
                    bucket_indices = self._bucket_xs(curr_model, debias_policy, xs, curr_coord)
                new_preds = old_preds

                # bucket indices will be a nested array
                for i in range(len(bucket_indices)):
                    bucket = bucket_indices[i]
                    if len(bucket)==0:
                        break
                    bias_term = self.bias_array[t][i]
                    new_preds[bucket]-= bias_term 
                
                # return new_preds
                return new_preds

            self.model = new_model
            return new_model
        
        def _halt(self, t):
            """
            Check if 0 bias in previous round; halts debiasing early if so.
            """

            # need to check for every possible coordinate, so need to check for the bias array
            # over the last pred_dim number of rounds. 
            if np.all(self.bias_array[t-self.policy.n_vals*self.policy.dim:t] == 0.0):
                return True
            else: return False
        
        def _truncate_bookkeeping(self, depth_reached):
            """
            helper function; if model debiasing has finished before max depth was reached, will truncate model accordingly.
            """
            self.training_buckets_indices = self.training_buckets_indices[:depth_reached]# store buckets by index of training data for each round
            self.training_buckets_preds = self.training_buckets_preds[:depth_reached]
            self.bias_array = self.bias_array[:depth_reached] # store bias terms for each round for each bucket
            self.training_preds = self.training_preds[:depth_reached+1] #also store preds of final model
            self.depth = depth_reached 
            return -1

    
    """
    Old ensembling code, but now we just do it on the fly during evaluation
    """

def run_ensemble(xs,ys,policies):
    pred_revs = []
    actual_revs = []
    for pi in policies:
        preds = pi.model(xs)
        allocation = pi.run_given_preds(preds)
        pred_rev = np.einsum('ij,ij->i', preds, allocation)
        actual_rev = np.einsum('ij,ij->i', ys, allocation)
        pred_revs.append(pred_rev)
        actual_revs.append(actual_rev)

    pred_revs = np.array(pred_revs)
    actual_revs = np.array(actual_revs)
    max_index = np.argmax(pred_revs, axis=0)
    max_pred_rev = pred_revs[max_index,np.arange(pred_revs.shape[1])]
    max_actual_rev = actual_revs[max_index]
    return [max_index, max_pred_rev, pred_revs, max_actual_rev, actual_revs]

def debias_all():
    return None


"""
Too spaghetti-y so I rewrote
"""

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

        To Do: Why did I implement this as halting when squared error drops, instead of just using the amount of bias??
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
    
    





            
            


