import numpy as np

class EnsembledModel:

    """
    "Max ensemble" version of debiasing. 
    
    Given a series of k initial models and their associated policies, it iteratively debiases all of them,
    conditioning on which policy induced by the models is maximal and a coord x val pair. 
    """

    def __init__(self, init_models, policies, train_x, train_y, max_depth):
        """
        init_models: An array of k different models for the prediction task. 
        policies: An array of k policies which are induced by those models. These should be policy objects.
        train_x: training data features
        train_y: training data labels
        max_depth: maximal depth of the debiasing process. 

        Currently, max_depth acts as the stopping condition for the model. If it is unbiased earlier, it will not halt. 
        """

        self.init_models = init_models
        self.policies = policies 
        self.train_x = train_x
        self.train_y = train_y
        self.max_depth = max_depth
        self.pred_dim = len(train_y[0]) # prediction dimension
        
        self.n_models = len(self.init_models)
        self.n_values = self.policies[0].n_vals
        self.policy_vals = self.policies[0].coordinate_values
        
        self.curr_depth = 0
        self.debias_conditions = []
        # keep current predictions *of all k models at once*
        self.curr_preds = [model(self.train_x) for model in self.init_models] 
        # keep track of bias array, which has k entries per round of debiasing
        self.bias_array = []
        self.predictions_by_round = [np.copy(self.curr_preds)]

    def debias(self):
        """
        W better data structure

        """
        for i in range(self.max_depth):
            

            # Get the conditions to debias with respect to in this round, and store.

            model_index = self.curr_depth % self.n_models 
            coord = (self.curr_depth//(self.n_values* self.n_models)) % self.pred_dim 
            val = self.policy_vals[(self.curr_depth // self.n_models) % self.n_values]
            
            self.debias_conditions.append([model_index, coord, val])

            # Evaluate models, determine the policy associated w each and that policy's self-assessed revenue
            policies_by_models = [self.policies[i].run_given_preds(self.curr_preds[i]) for i in range(self.n_models)]      
            self_assessed_revs =  np.array([np.einsum('ij,ij->i', self.curr_preds[i], policies_by_models[i]) for i in range(self.n_models)]) # array of length k, where each entry is of shape n, and is dot product of pred and policy vector
            maximal_policy = np.argmax(self_assessed_revs, axis=0) # length n; returns index of the maximal policy
            curr_policy = policies_by_models[model_index]
            bias_this_round= [] # will be k-dimensional, one entry for when each of the k policies is maximal
            
            for i in range(self.n_models):
                flag = (curr_policy[:,coord] == val) & (maximal_policy == i)
                if sum(flag)!=0:
                    bias = np.mean(self.curr_preds[model_index][flag] - self.train_y[flag], axis=0)
                    bias_this_round.append(bias)
                    self.curr_preds[model_index][flag] -= bias
                else:
                    # just store 0 if bucket is empty
                    bias_this_round.append(np.zeros(self.pred_dim))
            self.bias_array.append(np.array(bias_this_round))
            self.predictions_by_round.append(np.copy(self.curr_preds))

            self.curr_depth += 1    
    
    def predict(self, xs):
        """
        W better data structure
        Evaluate all k models in parallel
        """
        
        # evaluate all k initial models on the data
        curr_preds = [model(xs) for model in self.init_models]
        transcript = [np.copy(curr_preds)] # for plotting etc., output predictions over each round

        for i in range(self.curr_depth):
            # determine what the conditioning event for this round was
            model_index, coord, val = self.debias_conditions[i]
            # determine policy and policy's self assessed revenue for each model
            policies_by_models = [self.policies[i].run_given_preds(curr_preds[i]) for i in range(self.n_models)]      
            self_assessed_revs =  np.array([np.einsum('ij,ij->i', curr_preds[i], policies_by_models[i]) for i in range(self.n_models)])
            maximal_policy = np.argmax(self_assessed_revs, axis=0)

            # now, update the predictions according to each of the k debiasing events for that round
            curr_policy = policies_by_models[model_index]
            for k in range(self.n_models):
                flag = (curr_policy[:,coord] == val) & (maximal_policy == k)
                curr_preds[model_index][flag] -= self.bias_array[i][k] 
            
            transcript.append(np.copy(curr_preds))

        return curr_preds, transcript