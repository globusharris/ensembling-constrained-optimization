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