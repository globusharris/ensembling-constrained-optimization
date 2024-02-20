import numpy as np
import policies

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

    
    