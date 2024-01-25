import numpy as np
import policies

class Debiased_model:

    def __init__(self):
        self.depth = None #number of rounds of debiasing
        self.gran = None # granularity of level sets for debiasing (irrelevant for simplex policy)
        self.policy = None # policy used for optimization, should be a class
        self.prediction_dim = None #dimension of predictions
        self.bias_array = None #bookkeeping for bias terms 
        self.training_buckets_indices = None # bookkeeping for bucketing the training data per round of debiasing. stores indices of buckets.
        self.training_buckets_preds = None # bookkeeping for bucketing; stores predictions on training data so don't have to recompute for bias calc
        self.n = None #number of training datapoints
        self.model = None #(debiased) model generated

    def debias(self, X_train, Y_train, init_model, max_depth, policy, gran=None):
        self.prediction_dim = Y_train.shape[1]
        self.depth = max_depth
        self.policy = policy
        self.gran = gran
        self.n = len(X_train)
        self.model = init_model

        # bookkeeping
        n_buckets = self.policy.n_vals * self.policy.dim  #number of conditioning events that we'll bucket the data into 
        self.training_buckets_indices = [[[] for j in range(n_buckets)] for i in range(self.depth)] # store buckets by index of training data for each round
        self.training_buckets_preds = [[[] for j in range(n_buckets)] for i in range(self.depth)] # store buckets by value of predictions in each round (to avoid recomputing)
        self.bias_array = [[[] for j in range(n_buckets)] for i in range(self.depth)] # store bias terms for each round for each bucket

        # Begin debiasing process:
        t = 0

        while t < self.depth:
            print(t, 't')
            self._bucket_xs(self.model, X_train, t=t, save=True)
            self._update_bias_terms(t, Y_train)
            self._update_model(self.model, t)
            t += 1
        return self.model

    def _bucket_xs(self, curr_model, xs, t = None, save=False):
        """
        Helper function which takes the current model and training data, runs the current model
        on this data and then buckets the training data according to the policy's decisions on
        the current model. 

        curr_model: current predictive model
        xs: feature data that is being bucketed
        t: current round of debiasing (for bookkeeping, only needed if using in debiasing process)
        save: Boolean flag. If true, buckets will be stored globally for the training process. Otherwise, 
        bucketed indices are returned but not stored by the model. 

        Note: currently, this saves not the training x's themselves in a bucketed fashion but just their indices.
        This is hopefully a bit more versatile 
        """

        # get current model's predictions
        preds = curr_model(xs)
        # apply policy to each of the predictions 
        # for each of preds, this will generate a vector with length equal to self.prediction_length
        policies = self.policy.run(preds)  # dimension is len(X_train) x 
        # get the list of policy values that we care about for the conditioning events
        vals = self.policy.coordinate_values
        
        if not save:
            n_buckets = self.policy.n_vals * self.policy.dim 
            local_bucket_indices = [[] for i in range(n_buckets)]
                                    
        i = 0 # i iterates through coordinates of policy
        while i < self.policy.dim:
            j = 0
            while j < self.policy.n_vals: # j iterates through possible policy values 
                block_index = i * self.policy.n_vals 
                indices = np.arange(len(policies))[policies[:,i] == vals[j]]
                
                if save == True:
                    self.training_buckets_indices[t][block_index + j] = indices
                    self.training_buckets_preds[t][block_index + j] = preds[indices]
                else:
                    local_bucket_indices[block_index + j] = indices
                j += 1
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
                self.bias_array[t][i] = np.average(preds - ys, axis=0)
            else:
                # otherwise, never use this term, so not important, but for now filling w zeros
                # maybe shouldl use nans instead idk.
                self.bias_array[t][i] = np.zeros(self.prediction_dim)
        
        return self.bias_array
    
    def _update_model(self, curr_model, t):
        """
        There are kind of two ways of doing this. For the training data, I can do it more "statically", because
        I'm keeping track of everything along the way. But for any new data, will have to re-evaluate the model 
        at every round. For experiments not sure what kind of bookkeeping we'll want for the new data. 
        """
        def new_model(xs):
            old_preds = curr_model(xs)
            
            # print("old_preds", old_preds)
            bucket_indices = self._bucket_xs(curr_model, xs)
            new_preds = old_preds

            # bucket indices will be a nested array
            for i in range(len(bucket_indices)):
                bucket = bucket_indices[i]
                bias_term = self.bias_array[t][i]
                new_preds[bucket] = new_preds[bucket] - bias_term
            
            # return new_preds
            return new_preds

        self.model = new_model
        return new_model
    
    def _truncate_bias(self, depth_reached):
        # helper function; if model debiasing has finished before max depth was reached, will truncate model accordingly.
        return -1
    
    